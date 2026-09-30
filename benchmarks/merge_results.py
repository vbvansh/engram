"""Merge chunked benchmark runs into one baseline report.

A full 10-conversation pass takes many hours, so it is run in resumable chunks
(`--start-conv` / `--limit-convs`), each writing its own results directory.
This recombines those rows into the single number the project reports.

    python benchmarks/merge_results.py benchmarks/results/locomo-*
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_locomo import format_summary, summarize  # noqa: E402


def load_rows(paths: list[str]) -> list[dict]:
    """Read rows from every run directory, de-duplicating re-runs.

    The same (conversation, question) can appear in several chunks if a chunk
    was retried; the newest file wins so a re-run supersedes the attempt it
    replaced rather than being counted twice.
    """
    seen: dict[tuple, dict] = {}
    files = []
    for p in paths:
        path = Path(p)
        f = path / "rows.jsonl" if path.is_dir() else path
        if f.exists():
            files.append(f)
    for f in sorted(files, key=lambda x: x.stat().st_mtime):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen[(row.get("sample_id"), row.get("q_idx"))] = row
    return list(seen.values())


def main() -> int:
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob("benchmarks/results/locomo-*"))
    paths: list[str] = []
    for a in args:
        paths.extend(sorted(glob.glob(a)) or [a])
    rows = load_rows(paths)
    if not rows:
        print("no rows found in:", ", ".join(paths), file=sys.stderr)
        return 1

    s = summarize(rows)
    s["merged_from"] = paths
    print(format_summary(s, title="MERGED BASELINE"))
    print(f"\n  merged from {len(paths)} run dir(s), {len(s['by_conversation'])} conversation(s)")
    out = Path("benchmarks/results/merged_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
