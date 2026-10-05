"""LoCoMo baseline runner — the conductor.

Ties loader + engram_client + judge together into one benchmark pass:

    for each conversation:
        create a fresh isolated tenant            (Issue C: no cross-convo bleed)
        ingest every turn pair
        wait_for_drain()                          (Issue A: no async race)
        for each question:
            query Engram, judge the answer, record the FULL trace

Every question row keeps Engram's retrieval_metadata (route, answer mode, stop
reason, evidence counts, latency), not just right/wrong -- that trace is what
powers the later failure analysis without having to re-run the benchmark.

All tenants share one Neo4j vector index that is filtered by tenant only
after the nearest neighbours are found, so other conversations' memories can
crowd out results. For a fair baseline, store EVERY conversation first and ask
questions only afterwards, so each one is measured against the same index:

    # 1) store, conv-26 first (pilot: time + quota), then the other nine
    python benchmarks/run_locomo.py --ingest-only --tenant-run base1 --start-conv 0 --limit-convs 1
    python benchmarks/run_locomo.py --ingest-only --tenant-run base1 --start-conv 1

    # 2) ask all questions against the stored tenants
    python benchmarks/run_locomo.py --reuse-tenant "locomo-base1-c{conv}" --query-workers 3

    # 3) merge chunked question runs, if any
    python benchmarks/merge_results.py benchmarks/results/locomo-<run1> ...

Env (loaded from .env.local when run from the repo root):
    ENGRAM_BASE_URL       default http://127.0.0.1:8001
    ENGRAM_ADMIN_KEY      required (creates per-conversation tenants)
    ENGRAM_API_KEY        optional (records the server's model config)
    ENGRAM_DATABASE_URL   required (ingest progress is read from PostgreSQL)
    OPENCODE_GO_API_KEY   required (judge model; OPENCODE_API_KEY also works)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import httpx

# Allow running as `python benchmarks/run_locomo.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_quota import blocked_windows, fetch_usage, format_usage  # noqa: E402
from engram_client import DrainConfig, DrainTimeout, EngramClient, EngramError  # noqa: E402
from envfile import load_env_file  # noqa: E402
from judge import API_KEY_ENVS, DEFAULT_MODEL, OpenCodeJudge, judge_api_key  # noqa: E402
from loader import Conversation, load_locomo  # noqa: E402

# Row field -> summary section. merge_results.py uses the same groups.
GROUPS = {
    "by_category": "category_name",
    "by_route": "retrieval_route",
    "by_answer_mode": "answer_mode",
    "by_answerability": "answerability",
    "by_conversation": "sample_id",
}

_PACKAGES = ("neo4j", "openai", "psycopg", "sentence-transformers", "temporalio", "torch",
             "transformers")


def _session_pairs(conv: Conversation):
    """Yield (session_id, user_turn, assistant_turn_or_None) grouped per session.

    Pairs are formed WITHIN a session so the session_id and timestamp on each
    pair stay consistent and turns never pair across a session boundary. LoCoMo
    speakers alternate, so consecutive turns are a natural user/assistant pair;
    speaker identity is preserved inside the text via Turn.attributed(), so which
    slot a speaker lands in does not matter. A session with an odd turn count
    leaves a trailing turn paired with None (the caller pads it).
    """
    by_session: dict[int, list] = defaultdict(list)
    for t in conv.turns:
        by_session[t.session_idx].append(t)
    for sidx in sorted(by_session):
        turns = by_session[sidx]
        session_id = f"s{sidx}"
        for i in range(0, len(turns), 2):
            user = turns[i]
            asst = turns[i + 1] if i + 1 < len(turns) else None
            yield session_id, user, asst


def _dated(turn) -> str:
    """Turn text with an absolute date anchor prepended.

    LoCoMo utterances use relative time ("yesterday", "last year") and the real
    date lives only in the session timestamp. The canonical extractor receives
    no date, but this prefix survives in the stored conversation text, which
    lets the answer model resolve "yesterday" (the smoke test answered
    "7 May 2023" correctly this way).
    """
    if turn.timestamp:
        return f"[{turn.timestamp}] {turn.attributed()}"
    return turn.attributed()


def ingest_conversation(client: EngramClient, conv: Conversation, *, limit_pairs: int = 0) -> int:
    """Ingest every pair of one conversation. Returns the pair count.

    `limit_pairs` (>0) caps ingest for cheap plumbing tests; the baseline run
    leaves it at 0 (all pairs) so memory is complete.
    """
    n = 0
    for idx, (session_id, user, asst) in enumerate(_session_pairs(conv)):
        if limit_pairs and idx >= limit_pairs:
            break
        # Trailing lone turn -> empty assistant content (nothing extra to
        # extract), which keeps the user utterance in the record.
        asst_content = _dated(asst) if asst is not None else ""
        asst_ts = asst.timestamp if asst is not None else user.timestamp
        for attempt in range(3):
            try:
                client.ingest_pair(
                    session_id=session_id,
                    user_content=_dated(user),
                    assistant_content=asst_content,
                    user_timestamp=user.timestamp,
                    assistant_timestamp=asst_ts,
                    user_turn_idx=2 * idx,
                    assistant_turn_idx=2 * idx + 1,
                    source="locomo",
                )
                break
            except (EngramError, httpx.HTTPError):
                # A dropped connection or a 503 backpressure reply is transient.
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        n += 1
    return n


def load_question_list(path: str) -> dict[str, set[int]]:
    """Read a question list: one `sample_id:q_idx` per line (e.g. conv-26:12).

    Used for quick checks that re-ask only the questions a fix targets.
    Blank lines and lines starting with '#' are ignored.
    """
    selected: dict[str, set[int]] = defaultdict(set)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sample_id, _, q_idx = line.rpartition(":")
        selected[sample_id].add(int(q_idx))
    return dict(selected)


def run_config(base_url: str, judge_model: str) -> dict:
    """What produced this run: needed to compare runs honestly later."""
    repo = Path(__file__).resolve().parents[1]
    cfg: dict = {"judge_model": judge_model, "base_url": base_url}
    try:
        cfg["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        cfg["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        cfg["git_commit"] = None
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    cfg["packages"] = versions
    key = os.environ.get("ENGRAM_API_KEY")
    if key:
        try:
            resp = httpx.get(f"{base_url}/api/v1/config",
                             headers={"Authorization": f"Bearer {key}"}, timeout=30)
            if resp.status_code == 200:
                cfg["engram_models"] = resp.json()
        except httpx.HTTPError:
            pass
    return cfg


def _answer_one(client: EngramClient, judge: OpenCodeJudge, probe, max_depth: str | None):
    """Query Engram and judge the answer. Safe to call from worker threads."""
    started = time.monotonic()
    try:
        res = client.query(probe.question, max_depth=max_depth)
        answer = res.get("answer", "")
        meta = res.get("retrieval_metadata", {}) or {}
        answerability = res.get("answerability") or meta.get("answerability_state")
    except EngramError as err:
        answer, meta, answerability = f"(query-error: {err})", {}, "QUERY_ERROR"
    query_s = time.monotonic() - started
    jr = judge.judge(
        question=probe.question,
        gold_answer=probe.answer,
        predicted_answer=answer,
        is_adversarial=probe.is_adversarial,
    )
    return answer, meta, answerability, query_s, jr


def run(args: argparse.Namespace) -> int:
    env_used = load_env_file(args.env_file)
    if env_used:
        print(f"env: {env_used.resolve()}")
    base_url = args.base_url or os.environ.get("ENGRAM_BASE_URL", "http://127.0.0.1:8001")
    admin_key = args.admin_key or os.environ.get("ENGRAM_ADMIN_KEY")
    if not admin_key:
        print("error: set ENGRAM_ADMIN_KEY (needed to create tenants)", file=sys.stderr)
        return 2
    if not os.environ.get("ENGRAM_DATABASE_URL"):
        print("error: set ENGRAM_DATABASE_URL (ingest progress is read from PostgreSQL)",
              file=sys.stderr)
        return 2
    judge_key = judge_api_key()
    if not judge_key:
        print(f"error: set one of {', '.join(API_KEY_ENVS)} (needed for the judge)",
              file=sys.stderr)
        return 2

    # Pre-flight: a blown provider quota makes every ingest fail with 429, which
    # costs a long wait and yields nothing. Check before doing any work.
    if not args.skip_quota_check:
        try:
            usage = fetch_usage(judge_key)
            blocked = blocked_windows(usage)
            print("quota:")
            print(format_usage(usage))
            if blocked:
                print(f"\nerror: provider quota exhausted ({', '.join(blocked)}). "
                      f"Every model call will 429 — aborting before wasting time. "
                      f"Wait for reset or pass --skip-quota-check.", file=sys.stderr)
                return 3
        except Exception as err:  # non-fatal: never block a run on telemetry
            print(f"  (quota check unavailable: {err})")

    conversations = load_locomo(args.data)
    # Index conversations before slicing so a chunked run keeps stable ids.
    indexed = list(enumerate(conversations))
    if args.start_conv:
        indexed = indexed[args.start_conv:]
    if args.limit_convs:
        indexed = indexed[: args.limit_convs]

    run_id = datetime.now().strftime("%m%d%H%M%S")
    out_dir = Path(args.out) / f"locomo-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    summary_path = out_dir / "summary.json"

    drain_cfg = DrainConfig(max_wait_s=args.drain_timeout)
    print(f"run {run_id}: {len(indexed)} conversation(s) "
          f"[{indexed[0][0]}..{indexed[-1][0]}] -> {out_dir}" if indexed
          else f"run {run_id}: no conversations selected")

    selected = load_question_list(args.questions) if args.questions else None
    if selected is not None:
        print(f"selected questions: {sum(len(v) for v in selected.values())} "
              f"from {args.questions}")

    rows: list[dict] = []
    skipped: list[dict] = []
    ingests: dict[str, dict] = {}
    with OpenCodeJudge(model=args.judge_model) as judge, \
            open(rows_path, "w", encoding="utf-8") as rows_fh:
        for cidx, conv in indexed:
            if selected is not None and conv.sample_id not in selected:
                continue  # no selected question in this conversation
            # --reuse-tenant re-queries a tenant that was already ingested,
            # skipping the expensive ingest. It may be a literal id or a
            # template containing {conv} (tenants of a chunked run share a
            # run-id prefix but differ by conversation index).
            if args.reuse_tenant:
                tenant_id = args.reuse_tenant.replace("{conv}", str(cidx))
            else:
                tenant_id = f"{args.tenant_prefix}-{args.tenant_run or run_id}-c{cidx}"
            print(f"\n[conv {cidx}] {conv.sample_id} tenant={tenant_id}"
                  + ("  (reusing existing ingest)" if args.reuse_tenant else ""))
            try:
                client = EngramClient.create_tenant(
                    base_url=base_url, admin_key=admin_key,
                    tenant_id=tenant_id, display_name=conv.sample_id,
                    query_timeout_s=args.query_timeout,
                )
            except (EngramError, httpx.HTTPError) as err:
                print(f"  ! tenant setup failed, skipping conv: {err}")
                skipped.append({"conv_idx": cidx, "sample_id": conv.sample_id,
                                "reason": f"tenant setup failed: {err}"})
                continue

            with client:
                n_pairs = None
                if args.reuse_tenant:
                    print("  skipping ingest (--reuse-tenant)")
                else:
                    t0 = time.monotonic()
                    try:
                        n_pairs = ingest_conversation(client, conv, limit_pairs=args.limit_pairs)
                    except (EngramError, httpx.HTTPError) as err:
                        print(f"  ! ingest request failed, skipping conv: {err}")
                        skipped.append({"conv_idx": cidx, "sample_id": conv.sample_id,
                                        "reason": f"ingest request failed: {err}"})
                        continue
                    print(f"  sent {n_pairs} pairs in {time.monotonic() - t0:.1f}s; "
                          f"waiting for write + Neo4j copy...")

                def _progress(elapsed: float, events: dict, projections: dict) -> None:
                    done = sum(events.get(s, 0) for s in ("COMPLETE", "GATED_SKIP", "FAILED"))
                    total = sum(events.values()) or 1
                    copied = projections.get("COMPLETE", 0)
                    print(f"    ... {elapsed / 60:.1f}m  {done}/{total} messages written "
                          f"({100 * done / total:.0f}%), {copied}/{sum(projections.values())} "
                          f"neo4j copies done")

                drain = None
                try:
                    drain = client.wait_for_drain(drain_cfg, on_progress=_progress)
                    print(f"  drained in {drain.waited_s / 60:.1f}m | {drain.summary()}")
                    for reason, n in drain.failure_reasons.items():
                        print(f"    FAILED x{n}: {reason}")
                    ingests[conv.sample_id] = {**drain.to_dict(), "pairs_sent": n_pairs}
                except DrainTimeout as err:
                    print(f"  ! DRAIN FAILED: {err}")

                # Scoring a conversation whose ingest did not finish produces a
                # real-looking accuracy against a memory that was never built.
                # Skip it unless the operator explicitly opts in.
                if drain is None or not drain.healthy():
                    detail = drain.summary() if drain else "no drain result"
                    if not args.force:
                        print(f"  ! ingest incomplete ({detail}) — SKIPPING this "
                              f"conversation so it cannot produce a phantom score. "
                              f"Re-run with --force to score anyway.")
                        skipped.append({"conv_idx": cidx, "sample_id": conv.sample_id,
                                        "reason": detail})
                        continue
                    print(f"  ! ingest incomplete ({detail}) — scoring anyway (--force)")

                if args.ingest_only:
                    # Every conversation is stored before any question is asked,
                    # so all of them are queried against the same shared index.
                    print("  stored; questions skipped (--ingest-only)")
                    continue

                # Keep each question's ORIGINAL index: runs are compared
                # question by question on (sample_id, q_idx), so filtering must
                # never renumber them.
                questions = list(enumerate(conv.qa))
                if selected is not None:
                    questions = [(i, q) for i, q in questions
                                 if i in selected.get(conv.sample_id, set())]
                if args.categories:
                    wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
                    questions = [(i, q) for i, q in questions if q.category_name in wanted]
                if args.limit_questions:
                    questions = questions[: args.limit_questions]
                print(f"  asking {len(questions)} questions "
                      f"({args.query_workers} at a time)...")

                # Questions are independent reads, so they may run in parallel.
                # Results are written from this thread as they complete.
                with ThreadPoolExecutor(max_workers=args.query_workers) as pool:
                    futures = {
                        pool.submit(_answer_one, client, judge, probe, args.max_depth):
                            (qidx, probe)
                        for qidx, probe in questions
                    }
                    for future in as_completed(futures):
                        qidx, probe = futures[future]
                        answer, meta, answerability, query_s, jr = future.result()
                        row = {
                            "conv_idx": cidx,
                            "sample_id": conv.sample_id,
                            "tenant_id": tenant_id,
                            "q_idx": qidx,
                            "question": probe.question,
                            "category": probe.category,
                            "category_name": probe.category_name,
                            "is_adversarial": probe.is_adversarial,
                            "gold": probe.answer,
                            "evidence": probe.evidence,
                            "predicted": answer,
                            "correct": jr.correct,
                            "judged": jr.judged,
                            "judge_reason": jr.reason,
                            # raw judge text kept only when it failed, for debugging
                            "judge_raw": jr.raw if not jr.judged else "",
                            # --- retrieval trace (for failure analysis) ---
                            "answerability": answerability,
                            "retrieval_route": meta.get("retrieval_route"),
                            "routes_attempted": meta.get("routes_attempted"),
                            # The server leaves answer_mode unset when the answer
                            # model wrote the reply; name that case explicitly.
                            "answer_mode": meta.get("answer_mode") or (
                                "llm" if meta else "error"),
                            "stop_reason": meta.get("stop_reason"),
                            "candidates_discovered": meta.get("candidates_discovered"),
                            "verified_evidence": meta.get("verified_evidence"),
                            "missing_evidence": meta.get("missing_evidence"),
                            "total_context_tokens": meta.get("total_context_tokens"),
                            "query_s": round(query_s, 1),
                            "retrieval_metadata": meta,
                        }
                        rows.append(row)
                        rows_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows_fh.flush()

                        mark = "OK " if jr.correct else "XX "
                        if qidx % 10 == 0 or not jr.correct:
                            print(f"    [{mark}] q{qidx} {probe.category_name:>11} "
                                  f"route={row['retrieval_route']} mode={row['answer_mode']}"
                                  f" :: {probe.question[:60]}")

    summary = summarize(rows)
    summary["run_id"] = run_id
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["config"] = run_config(base_url, args.judge_model)
    summary["ingest"] = ingests
    summary["skipped_conversations"] = skipped
    summary["mode"] = ("ingest-only" if args.ingest_only
                       else "questions-only" if args.reuse_tenant else "ingest+questions")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.ingest_only:
        print("\n" + "=" * 62 + "\nINGEST ONLY - no questions asked\n" + "=" * 62)
        for sample_id, info in ingests.items():
            print(f"  {sample_id}: {info['pairs_sent']} pairs in {info['waited_s'] / 60:.1f}m"
                  f" | events={info['events']} neo4j copies={info['projections']}")
            for reason, n in info["failure_reasons"].items():
                print(f"      FAILED x{n}: {reason}")
        for s in skipped:
            print(f"  ! SKIPPED conv{s['conv_idx']} ({s['sample_id']}): {s['reason']}")
    else:
        print(format_summary(summary, title="RUN"))
    print(f"\nrows:    {rows_path}")
    print(f"summary: {summary_path}")
    return 0


def _acc(rows: list[dict]) -> dict:
    n = len(rows)
    c = sum(1 for r in rows if r.get("correct"))
    return {"n": n, "correct": c, "accuracy": (c / n if n else 0.0)}


def summarize(rows: list[dict]) -> dict:
    out: dict = {"overall": _acc(rows)}
    for name, key in GROUPS.items():
        groups: dict[str, list] = defaultdict(list)
        for r in rows:
            groups[str(r.get(key))].append(r)
        out[name] = {k: _acc(v) for k, v in sorted(groups.items())}
    out["judge_failures"] = sum(1 for r in rows if r.get("judged") is False)
    return out


def format_summary(summary: dict, *, title: str) -> str:
    o = summary["overall"]
    lines = ["", "=" * 62, f"{title}  accuracy = {o['accuracy']:.1%}  ({o['correct']}/{o['n']})",
             "=" * 62]
    for name in GROUPS:
        lines.append(f"\n{name.replace('_', ' ')}")
        for k, v in summary.get(name, {}).items():
            lines.append(f"  {k:>28}: {v['accuracy']:>6.1%}  ({v['correct']}/{v['n']})")
    if summary.get("judge_failures"):
        lines.append(f"\n  ! {summary['judge_failures']} judge failure(s) — flagged for re-judging")
    for s in summary.get("skipped_conversations", []):
        lines.append(f"\n  ! SKIPPED conv{s['conv_idx']} ({s['sample_id']}): {s['reason']}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Run the LoCoMo baseline against Engram.")
    p.add_argument("--data", default="benchmarks/data/locomo10.json")
    p.add_argument("--base-url", default=None)
    p.add_argument("--admin-key", default=None)
    p.add_argument("--limit-convs", type=int, default=0, help="0 = all")
    p.add_argument("--start-conv", type=int, default=0,
                   help="0-based index of the first conversation to run; use with "
                        "--limit-convs to process the benchmark in resumable chunks")
    p.add_argument("--limit-questions", type=int, default=0, help="0 = all per conv")
    p.add_argument("--questions", default=None, metavar="FILE",
                   help="ask only the questions listed in FILE (one sample_id:q_idx per "
                        "line), e.g. the ones a fix targets")
    p.add_argument("--categories", default="",
                   help="comma-separated category filter, e.g. 'temporal' "
                        "(names: multi_hop,temporal,open_domain,single_hop,adversarial)")
    p.add_argument("--limit-pairs", type=int, default=0,
                   help="0 = all; cap ingested pairs for a cheap plumbing test")
    p.add_argument("--max-depth", default=None, help="cap retrieval route, e.g. L2")
    p.add_argument("--drain-timeout", type=float, default=10800.0,
                   help="max wait (s) for one conversation's messages to be written "
                        "and copied to Neo4j")
    p.add_argument("--query-timeout", type=float, default=300.0,
                   help="per-query HTTP timeout (s); queries drive several LLM calls")
    p.add_argument("--query-workers", type=int, default=1,
                   help="questions asked in parallel (answers do not depend on each other)")
    p.add_argument("--judge-model", default=DEFAULT_MODEL,
                   help="keep fixed across every run you want to compare")
    p.add_argument("--force", action="store_true",
                   help="score a conversation even if its ingest did not complete "
                        "(off by default so broken runs cannot yield phantom scores)")
    p.add_argument("--env-file", default=None,
                   help="path to the env file (auto: ./.env.local, ./.env, ../.env)")
    p.add_argument("--skip-quota-check", action="store_true",
                   help="run even if the provider reports an exhausted quota")
    p.add_argument("--reuse-tenant", default=None, metavar="TENANT_OR_TEMPLATE",
                   help="query an already-ingested tenant instead of re-ingesting; "
                        "may contain {conv}, e.g. locomo-0930120000-c{conv}")
    p.add_argument("--ingest-only", action="store_true",
                   help="store and wait, but ask no questions; ask them later with "
                        "--reuse-tenant once every conversation is stored")
    p.add_argument("--tenant-run", default=None, metavar="NAME",
                   help="fixed middle part of tenant ids (locomo-NAME-c0...) so chunked "
                        "ingest runs share one naming scheme; default: this run's id")
    p.add_argument("--tenant-prefix", default="locomo")
    p.add_argument("--out", default="benchmarks/results")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
