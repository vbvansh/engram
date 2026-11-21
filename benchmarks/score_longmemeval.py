"""Grade a LongMemEval run exactly like the benchmark's official evaluation.

Reproduces LongMemEval src/evaluation/evaluate_qa.py + print_qa_metrics.py:
the judge is gpt-4o-2024-08-06 on OpenAI (temperature 0, max_tokens 10), the
prompt depends on the question type (abstention questions, id ending "_abs",
get their own), and an answer is correct iff "yes" is in the judge's reply.
Reported like print_qa_metrics.py: accuracy per question type, task-averaged
accuracy, overall accuracy and abstention accuracy.

    python benchmarks/score_longmemeval.py benchmarks/results/longmemeval-<run>

Writes hypotheses.jsonl.eval-results-gpt-4o next to the run, in the official
format (each hypothesis line plus autoeval_label {model, label}), so the
official print_qa_metrics.py can read it too. Verdicts are cached there: a
re-run only grades new or failed answers. Needs OPENAI_API_KEY (Code\\.env).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envfile import load_env_file  # noqa: E402
from judge import LONGMEMEVAL_OFFICIAL_JUDGE, official_judge  # noqa: E402
from run_longmemeval import QUESTION_TYPES  # noqa: E402


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metrics(rows: list[dict]) -> dict:
    """print_qa_metrics.py: per-type, task-averaged, overall, abstention accuracy."""
    graded = [r for r in rows if r.get("autoeval_label", {}).get("label") is not None]
    per_type = {t: [float(r["autoeval_label"]["label"]) for r in graded if r["question_type"] == t]
                for t in QUESTION_TYPES}
    task_means = [_mean(v) for v in per_type.values() if v]
    abstention = [float(r["autoeval_label"]["label"]) for r in graded if r["abstention"]]
    return {
        "graded": len(graded),
        "by_type": {t: {"accuracy": _mean(v), "n": len(v)} for t, v in per_type.items()},
        "task_averaged_accuracy": _mean([m for m in task_means if m is not None]),
        "overall_accuracy": _mean([float(r["autoeval_label"]["label"]) for r in graded]),
        "abstention_accuracy": _mean(abstention),
        "abstention_n": len(abstention),
    }


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}"


def main() -> int:
    p = argparse.ArgumentParser(description="Official LongMemEval grading (gpt-4o judge).")
    p.add_argument("run", help="run directory with rows.jsonl")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    load_env_file()

    run_dir = Path(args.run)
    rows = [json.loads(line) for line in (run_dir / "rows.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    results_path = run_dir / "hypotheses.jsonl.eval-results-gpt-4o"
    cache: dict[tuple, dict] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            label = old.get("autoeval_label", {})
            if label.get("model") == LONGMEMEVAL_OFFICIAL_JUDGE and label.get("label") is not None:
                cache[(old["question_id"], old["hypothesis"])] = label
    for r in rows:
        if (r["question_id"], r["hypothesis"]) in cache:
            r["autoeval_label"] = cache[(r["question_id"], r["hypothesis"])]

    todo = [r for r in rows if "autoeval_label" not in r]
    print(f"official judge {LONGMEMEVAL_OFFICIAL_JUDGE}: {len(todo)} to grade, "
          f"{len(rows) - len(todo)} cached")
    if todo:
        with official_judge(LONGMEMEVAL_OFFICIAL_JUDGE) as judge:
            def grade(r: dict) -> None:
                res = judge.judge_longmemeval(
                    question_type=r["question_type"], question=r["question"],
                    answer=str(r["answer"]), response=r["hypothesis"], abstention=r["abstention"],
                )
                r["autoeval_label"] = {
                    "model": LONGMEMEVAL_OFFICIAL_JUDGE,
                    "label": res.correct if res.judged else None,
                    "raw": res.raw.strip(),
                }

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for f in as_completed([pool.submit(grade, r) for r in todo]):
                    f.result()

    with open(results_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"question_id": r["question_id"], "hypothesis": r["hypothesis"],
                                 "autoeval_label": r["autoeval_label"]}, ensure_ascii=False) + "\n")
    m = metrics(rows)
    (run_dir / "scores.json").write_text(json.dumps({**m, "judge": LONGMEMEVAL_OFFICIAL_JUDGE},
                                                    indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"LongMemEval official metrics (judge {LONGMEMEVAL_OFFICIAL_JUDGE}), "
          f"{m['graded']}/{len(rows)} graded")
    print("=" * 70)
    print("Evaluation results by task:")
    for t, v in m["by_type"].items():
        print(f"\t{t}: {_pct(v['accuracy'])} ({v['n']})")
    print(f"\nTask-averaged Accuracy: {_pct(m['task_averaged_accuracy'])}")
    print(f"Overall Accuracy: {_pct(m['overall_accuracy'])}")
    print(f"Abstention Accuracy: {_pct(m['abstention_accuracy'])} ({m['abstention_n']})")
    failed = sum(1 for r in rows if r["autoeval_label"].get("label") is None)
    if failed:
        print(f"! {failed} judge failure(s): re-run this command to retry them")
    print(f"\nresults: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
