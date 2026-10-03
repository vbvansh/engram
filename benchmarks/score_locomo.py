"""Re-score saved LoCoMo answers with the metrics papers report.

Engram's answers are already in rows.jsonl, so scoring never needs a re-run:

* F1  -- the original LoCoMo metric (Maharana et al., ACL 2024), computed
         exactly like snap-research/locomo task_eval/evaluation.py: normalise,
         Porter-stem, token-overlap F1; multi-hop (cat 1) averages per-part F1
         over comma-split sub-answers; open-domain (cat 3) keeps the gold text
         before ';'; adversarial (cat 5) scores 1 only if the answer contains
         "no information available" or "not mentioned".
* B1  -- BLEU-1 (unigram BLEU, smoothing method1), as reported by Mem0.
* J   -- LLM-as-a-Judge with the Mem0 paper's prompt (lenient, binary), on
         categories 1-4 only (1,540 questions), as in the Mem0 paper. Needs
         --judge (one model call per question); results are cached in
         scored.jsonl so an interrupted run resumes.

Our own strict judge (the `correct` field written by run_locomo.py) is kept
alongside: it stays the internal before/after metric, because the lenient J
prompt accepts "right topic, wrong specifics" answers.

    python benchmarks/score_locomo.py benchmarks/results/locomo-<run>
    python benchmarks/score_locomo.py benchmarks/results/locomo-<run> --judge --workers 6

Needs nltk (Porter stemmer and BLEU): pip install nltk
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nltk.stem import PorterStemmer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envfile import load_env_file  # noqa: E402
from judge import DEFAULT_MODEL, OpenCodeJudge  # noqa: E402
from merge_results import load_rows  # noqa: E402

_stemmer = PorterStemmer()
_smooth = SmoothingFunction().method1
STANDARD_CATEGORIES = ("single_hop", "multi_hop", "temporal", "open_domain")


# --- LoCoMo F1 (port of snap-research/locomo task_eval/evaluation.py) ---------


def _normalize_answer(s: str) -> str:
    s = s.replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred = [_stemmer.stem(w) for w in _normalize_answer(prediction).split()]
    gold = [_stemmer.stem(w) for w in _normalize_answer(ground_truth).split()]
    same = sum((Counter(pred) & Counter(gold)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(gold)
    return 2 * precision * recall / (precision + recall)


def _f1_multi(prediction: str, ground_truth: str) -> float:
    preds = [p.strip() for p in prediction.split(",")]
    golds = [g.strip() for g in ground_truth.split(",")]
    return sum(max(_f1_score(p, g) for p in preds) for g in golds) / len(golds)


def locomo_f1(prediction: str, gold: str, category: int) -> float:
    gold = str(gold)
    if category == 3:
        gold = gold.split(";")[0].strip()
    if category in (2, 3, 4):
        return _f1_score(prediction, gold)
    if category == 1:
        return _f1_multi(prediction, gold)
    if category == 5:
        low = prediction.lower()
        return 1.0 if ("no information available" in low or "not mentioned" in low) else 0.0
    raise ValueError(f"unknown LoCoMo category {category}")


def bleu1(prediction: str, gold: str) -> float:
    pred = re.findall(r"\w+", prediction.lower())
    ref = re.findall(r"\w+", str(gold).lower())
    if not pred or not ref:
        return 0.0
    return sentence_bleu([ref], pred, weights=(1, 0, 0, 0), smoothing_function=_smooth)


# --- scoring ---------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict]) -> dict:
    std = [r for r in rows if r["category_name"] in STANDARD_CATEGORIES]
    adv = [r for r in rows if r["category_name"] == "adversarial"]
    judged_j = [r for r in std if r.get("j_mem0") is not None]

    def block(group: list[dict]) -> dict:
        jj = [r for r in group if r.get("j_mem0") is not None]
        return {
            "n": len(group),
            "J_mem0": round(100 * _mean([float(r["j_mem0"]) for r in jj]), 2) if jj else None,
            "F1": round(100 * _mean([r["f1"] for r in group]), 2),
            "BLEU1": round(100 * _mean([r["bleu1"] for r in group]), 2),
            "strict_judge": round(100 * _mean([float(bool(r["correct"])) for r in group]), 2),
        }

    by_cat = defaultdict(list)
    for r in std:
        by_cat[r["category_name"]].append(r)
    return {
        "standard_1540": block(std),
        "standard_by_category": {k: block(v) for k, v in sorted(by_cat.items())},
        "adversarial": {
            "n": len(adv),
            "locomo_rule": round(100 * _mean([r["f1"] for r in adv]), 2),
            "strict_judge": round(100 * _mean([float(bool(r["correct"])) for r in adv]), 2),
        },
        "all_1986_strict_judge": round(100 * _mean([float(bool(r["correct"])) for r in rows]), 2),
        "j_mem0_judged": len(judged_j),
        "j_mem0_failures": sum(1 for r in std if r.get("j_mem0_failed")),
    }


def print_summary(s: dict, judge_model: str | None) -> None:
    std = s["standard_1540"]
    print("\n" + "=" * 70)
    print(f"STANDARD LoCoMo view: categories 1-4, n={std['n']} (adversarial excluded)")
    print("=" * 70)
    j = "-" if std["J_mem0"] is None else f"{std['J_mem0']:.2f}"
    print(f"  J (Mem0 judge prompt{', ' + judge_model if judge_model else ''}): {j}")
    print(f"  F1 (LoCoMo official):  {std['F1']:.2f}")
    print(f"  BLEU-1:                {std['BLEU1']:.2f}")
    print(f"  our strict judge:      {std['strict_judge']:.2f}")
    print(f"\n  {'category':<12} {'n':>5} {'J':>7} {'F1':>7} {'B1':>7} {'strict':>7}")
    for cat, b in s["standard_by_category"].items():
        jv = "-" if b["J_mem0"] is None else f"{b['J_mem0']:.2f}"
        print(f"  {cat:<12} {b['n']:>5} {jv:>7} {b['F1']:>7.2f} {b['BLEU1']:>7.2f} "
              f"{b['strict_judge']:>7.2f}")
    a = s["adversarial"]
    print(f"\n  adversarial (n={a['n']}, reported separately): LoCoMo string rule "
          f"{a['locomo_rule']:.2f} | our strict judge {a['strict_judge']:.2f}")
    print(f"  all 1,986 questions, our strict judge: {s['all_1986_strict_judge']:.2f}")
    if s["j_mem0_failures"]:
        print(f"  ! {s['j_mem0_failures']} J judge failure(s) — re-run with --judge to retry them")


def main() -> int:
    p = argparse.ArgumentParser(description="Re-score saved LoCoMo answers (F1, BLEU-1, J).")
    p.add_argument("runs", nargs="+", help="run directories (or rows.jsonl files) to score")
    p.add_argument("--judge", action="store_true",
                   help="also compute J with the Mem0 prompt (one model call per question)")
    p.add_argument("--judge-model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=6, help="parallel judge calls")
    p.add_argument("--out", default=None,
                   help="output directory (default: the first run directory)")
    args = p.parse_args()
    load_env_file()

    rows = load_rows(args.runs)
    if not rows:
        print("no rows found", file=sys.stderr)
        return 1
    out_dir = Path(args.out or (args.runs[0] if Path(args.runs[0]).is_dir()
                                else Path(args.runs[0]).parent))
    scored_path = out_dir / "scored.jsonl"

    # Reuse earlier J verdicts (resume) when the same answer was already judged.
    cache: dict[tuple, dict] = {}
    if scored_path.exists():
        for line in scored_path.read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            cache[(old["sample_id"], old["q_idx"], old["predicted"])] = old

    for r in rows:
        r["f1"] = locomo_f1(r["predicted"], r["gold"], int(r["category"]))
        r["bleu1"] = bleu1(r["predicted"], r["gold"])
        old = cache.get((r["sample_id"], r["q_idx"], r["predicted"]), {})
        for key in ("j_mem0", "j_mem0_reason", "j_mem0_failed", "strict_rejudged"):
            if key in old:
                r[key] = old[key]
        if old.get("strict_rejudged"):
            r["correct"], r["judged"] = old["correct"], old["judged"]

    if args.judge:
        with OpenCodeJudge(model=args.judge_model) as judge:
            todo = [r for r in rows if r["category_name"] in STANDARD_CATEGORIES
                    and (r.get("j_mem0") is None or r.get("j_mem0_failed"))]
            retry = [r for r in rows if r.get("judged") is False]
            print(f"J (Mem0 prompt): {len(todo)} to judge; strict re-judge of "
                  f"{len(retry)} earlier judge failure(s)")

            def run_j(r: dict) -> None:
                res = judge.judge_mem0(question=r["question"], gold_answer=str(r["gold"]),
                                       predicted_answer=r["predicted"])
                r["j_mem0"] = res.correct if res.judged else None
                r["j_mem0_reason"] = res.reason
                r["j_mem0_failed"] = not res.judged

            def run_strict(r: dict) -> None:
                res = judge.judge(question=r["question"], gold_answer=str(r["gold"]),
                                  predicted_answer=r["predicted"],
                                  is_adversarial=r["is_adversarial"])
                if res.judged:
                    r["correct"], r["judged"], r["judge_reason"] = res.correct, True, res.reason
                    r["strict_rejudged"] = True

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_j, r) for r in todo]
                futures += [pool.submit(run_strict, r) for r in retry]
                for i, f in enumerate(as_completed(futures), 1):
                    f.result()
                    if i % 100 == 0 or i == len(futures):
                        print(f"  ... {i}/{len(futures)} judged")

    with open(scored_path, "w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda x: (x["sample_id"], x["q_idx"])):
            slim = {k: v for k, v in r.items() if k != "retrieval_metadata"}
            fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    summary["judge_model"] = args.judge_model if args.judge else None
    summary["runs"] = args.runs
    (out_dir / "scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, args.judge_model if args.judge or summary["j_mem0_judged"] else None)
    print(f"\nscored rows: {scored_path}\nscores:      {out_dir / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
