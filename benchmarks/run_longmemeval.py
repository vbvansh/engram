"""Run LongMemEval (Wu et al., ICLR 2025) against Engram's HTTP API.

Data: the authors' cleaned release (2025/09), LongMemEval_S:

    benchmarks/data/longmemeval/longmemeval_s_cleaned.json
    (huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)

Every question has its own chat history (~48 sessions, ~490 turns), so every
question gets its own tenant. For each selected question the history is
stored session by session in date order (each turn carries its session's
timestamp), the store is waited out, and then the question is asked with
"Current Date: <question_date>" given to Engram's answer step only (memory
search uses the question alone).

Output (benchmarks/results/longmemeval-<run>/):
    rows.jsonl         one row per question: answer + retrieval trace
    hypotheses.jsonl   {question_id, hypothesis} -- the official input format
    selected.txt       the question ids, so another system gets the same sample

Grade with the official judge afterwards (score_longmemeval.py).

    # pilot: one question per type (6), stored then asked
    python benchmarks/run_longmemeval.py --per-type 1 --tenant-run lmep1
    # the same questions later, e.g. on another Engram, from the saved list
    python benchmarks/run_longmemeval.py --tenant-run other1 \
        --question-ids benchmarks/results/longmemeval-<run>/selected.txt

Re-running the same command resumes: pairs already stored are no-ops, and
--retry-failed sends messages that failed (e.g. in a provider outage) back
through Engram's retry API.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engram_client import DrainConfig, DrainTimeout, EngramClient, EngramError  # noqa: E402
from envfile import load_env_file  # noqa: E402
from run_locomo import run_config  # noqa: E402

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)
MAX_TURN_CHARS = 32_000  # Engram's per-turn limit (4 of 246,750 LongMemEval turns are longer)


def lme_timestamp(raw: str) -> str:
    """'2023/05/20 (Sat) 02:21' -> '2023-05-20T02:21:00'."""
    day, _, rest = raw.partition(" (")
    clock = rest.partition(") ")[2] or "00:00"
    return datetime.strptime(f"{day} {clock}", "%Y/%m/%d %H:%M").isoformat()


def session_pairs(turns: list[dict]) -> list[tuple[str, str]]:
    """(user, assistant) pairs in order; a turn without its partner gets an empty one."""
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(turns):
        if turns[i]["role"] == "user":
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            partner = nxt if nxt is not None and nxt["role"] == "assistant" else None
            pairs.append((turns[i]["content"], partner["content"] if partner else ""))
            i += 2 if partner else 1
        else:
            pairs.append(("", turns[i]["content"]))
            i += 1
    return [(u[:MAX_TURN_CHARS], a[:MAX_TURN_CHARS]) for u, a in pairs]


def select_questions(
    data: list[dict], per_type: int, seed: int, ids_file: str | None
) -> list[dict]:
    if ids_file:
        wanted = [line.strip() for line in Path(ids_file).read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.startswith("#")]
        by_id = {q["question_id"]: q for q in data}
        return [by_id[qid] for qid in wanted]
    rng = random.Random(seed)
    chosen: list[dict] = []
    for qtype in QUESTION_TYPES:
        pool = sorted(
            (q for q in data if q["question_type"] == qtype), key=lambda q: q["question_id"]
        )
        chosen += rng.sample(pool, min(per_type, len(pool)))
    return chosen


def store_history(client: EngramClient, question: dict) -> tuple[int, int]:
    """Send every session of one question's history. Returns (pairs, truncated turns)."""
    sent = truncated = 0
    for session_id, date, turns in zip(question["haystack_session_ids"], question["haystack_dates"],
                                       question["haystack_sessions"]):
        stamp = lme_timestamp(date)
        truncated += sum(len(t["content"]) > MAX_TURN_CHARS for t in turns)
        for k, (user, assistant) in enumerate(session_pairs(turns)):
            for attempt in range(3):
                try:
                    client.ingest_pair(
                        session_id=session_id, user_content=user, assistant_content=assistant,
                        user_timestamp=stamp, assistant_timestamp=stamp,
                        user_turn_idx=2 * k, assistant_turn_idx=2 * k + 1, source="longmemeval",
                    )
                    break
                except (EngramError, httpx.HTTPError):
                    # A dropped connection or a 503 backpressure reply is transient.
                    if attempt == 2:
                        raise
                    time.sleep(5 * (attempt + 1))
            sent += 1
    return sent, truncated


def _progress(elapsed: float, events: dict, _projections: dict) -> None:
    done = sum(events.get(s, 0) for s in ("COMPLETE", "GATED_SKIP", "FAILED"))
    print(f"    ... {elapsed / 60:.1f}m {done}/{sum(events.values())} messages done")


def run(args: argparse.Namespace) -> int:
    env_used = load_env_file(args.env_file)
    if env_used:
        print(f"env: {env_used.resolve()}")
    base_url = os.environ.get("ENGRAM_BASE_URL", "http://127.0.0.1:8001")
    admin_key = os.environ.get("ENGRAM_ADMIN_KEY")
    if not admin_key:
        print("error: ENGRAM_ADMIN_KEY is required", file=sys.stderr)
        return 2
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    questions = select_questions(data, args.per_type, args.seed, args.question_ids)
    run_id = datetime.now().strftime("%m%d%H%M%S")
    out_dir = Path(args.out) / f"longmemeval-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected.txt").write_text(
        "# LongMemEval question ids of this run (one per line)\n"
        + "\n".join(q["question_id"] for q in questions) + "\n", encoding="utf-8")
    print(f"run {run_id}: {len(questions)} question(s) "
          f"{dict(Counter(q['question_type'] for q in questions))} -> {out_dir}")

    drain_cfg = DrainConfig(max_wait_s=args.drain_timeout)
    rows: list[dict] = []
    stopped = None
    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as rows_fh, \
            open(out_dir / "hypotheses.jsonl", "w", encoding="utf-8") as hyp_fh:
        for n, q in enumerate(questions, 1):
            qid = q["question_id"]
            tenant_id = f"lme-{args.tenant_run}-{qid}".lower()
            print(f"\n[{n}/{len(questions)}] {qid} ({q['question_type']}) tenant={tenant_id}")
            try:
                client = EngramClient.create_tenant(
                    base_url=base_url, admin_key=admin_key, tenant_id=tenant_id,
                    display_name=qid, query_timeout_s=args.query_timeout,
                )
            except (EngramError, httpx.HTTPError) as err:
                print(f"  ! tenant setup failed: {err}")
                stopped = f"tenant setup failed at {qid}"
                break
            with client:
                ingest = None
                if not args.skip_ingest:
                    t0 = time.monotonic()
                    pairs, truncated = store_history(client, q)
                    print(f"  sent {pairs} pairs from {len(q['haystack_sessions'])} sessions in "
                          f"{time.monotonic() - t0:.0f}s"
                          + (f" ({truncated} over-long turn(s) cut to {MAX_TURN_CHARS} chars)"
                             if truncated else ""))
                    if args.retry_failed:
                        try:
                            print(f"  retried {client.retry_failed_events()} failed message(s)")
                        except (EngramError, httpx.HTTPError, psycopg.Error) as err:
                            print(f"  ! retrying failed messages did not work: {err}")
                    try:
                        drain = client.wait_for_drain(drain_cfg, on_progress=_progress)
                    except DrainTimeout as err:
                        print(f"  ! DRAIN FAILED: {err}")
                        stopped = (f"store stalled at {qid}; fix the stack, then re-run the same "
                                   f"command with --retry-failed")
                        break
                    print(f"  stored in {drain.waited_s / 60:.1f}m | {drain.summary()}")
                    ingest = {**drain.to_dict(), "pairs_sent": pairs, "turns_truncated": truncated}
                if args.ingest_only:
                    continue
                started = time.monotonic()
                try:
                    res = client.query(q["question"],
                                       answer_instructions=f"Current Date: {q['question_date']}")
                    hypothesis = res.get("answer", "")
                    meta = res.get("retrieval_metadata", {}) or {}
                except EngramError as err:
                    hypothesis, meta = f"(query-error: {err})", {}
                row = {
                    "question_id": qid,
                    "question_type": q["question_type"],
                    "abstention": qid.endswith("_abs"),
                    "question": q["question"],
                    "question_date": q["question_date"],
                    "answer": q["answer"],
                    "hypothesis": hypothesis,
                    "tenant_id": tenant_id,
                    "answer_session_ids": q["answer_session_ids"],
                    "query_s": round(time.monotonic() - started, 1),
                    "answer_mode": meta.get("answer_mode") or ("llm" if meta else "error"),
                    "routes_attempted": meta.get("routes_attempted"),
                    "verified_evidence": meta.get("verified_evidence"),
                    "ingest": ingest,
                    "retrieval_metadata": meta,
                }
                rows.append(row)
                rows_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_fh.flush()
                hyp_fh.write(json.dumps({"question_id": qid, "hypothesis": hypothesis},
                                        ensure_ascii=False) + "\n")
                hyp_fh.flush()
                print(f"  answer ({row['query_s']}s): {hypothesis[:160]}")

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "tenant_run": args.tenant_run,
        "questions": len(questions),
        "answered": len(rows),
        "by_type": dict(Counter(r["question_type"] for r in rows)),
        "query_errors": sum(r["hypothesis"].startswith("(query-error") for r in rows),
        "stopped": stopped,
        "config": run_config(base_url, "none (grade with score_longmemeval.py)"),
        "mode": "ingest-only" if args.ingest_only else "questions-only" if args.skip_ingest
        else "ingest+questions",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    status = f"STOPPED: {stopped}" if stopped else "done"
    print(f"\n{status} | answered {len(rows)}/{len(questions)} | rows: {out_dir / 'rows.jsonl'}")
    if stopped:
        print(f"  resume: same command plus --retry-failed (tenant-run {args.tenant_run})")
    return 1 if stopped else 0


def main() -> int:
    p = argparse.ArgumentParser(description="Run LongMemEval against Engram.")
    p.add_argument("--data", default="benchmarks/data/longmemeval/longmemeval_s_cleaned.json")
    p.add_argument("--per-type", type=int, default=1,
                   help="questions per question type, drawn with --seed (6 types)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--question-ids", default=None,
                   help="file with question ids (e.g. an earlier run's selected.txt)")
    p.add_argument("--tenant-run", required=True,
                   help="name in the tenant ids (lme-<name>-<question_id>); reuse it to resume")
    p.add_argument("--ingest-only", action="store_true", help="store the histories, ask nothing")
    p.add_argument("--skip-ingest", action="store_true",
                   help="ask only (the histories of this --tenant-run are already stored)")
    p.add_argument("--retry-failed", action="store_true",
                   help="send each question's FAILED messages back through Engram's retry API")
    p.add_argument("--query-timeout", type=float, default=300.0)
    p.add_argument("--drain-timeout", type=float, default=10800.0)
    p.add_argument("--out", default="benchmarks/results")
    p.add_argument("--env-file", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
