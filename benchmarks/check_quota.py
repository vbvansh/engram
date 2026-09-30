"""Check OpenCode Zen Go quota with only the API key (no dashboard login).

GET /zen/go/v1/usage reports three nested windows — rolling (5h), weekly, and
monthly. Any one of them at 100% blocks inference, which surfaces as HTTP 429
mid-run. The quota is shared across the workspace, so other people's usage
moves it too. Run it before starting a long benchmark:

    python benchmarks/check_quota.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envfile import load_env_file  # noqa: E402
from judge import API_KEY_ENVS, DEFAULT_BASE, judge_api_key  # noqa: E402

USAGE_URL = DEFAULT_BASE.rstrip("/") + "/usage"


def fetch_usage(api_key: str, *, timeout_s: float = 20.0) -> dict:
    resp = httpx.get(USAGE_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json().get("usage", {})


def blocked_windows(usage: dict) -> list[str]:
    """Windows currently blocking inference (100% / rate-limited)."""
    blocked = []
    for name, info in usage.items():
        if not isinstance(info, dict):
            continue
        if info.get("status") == "rate-limited" or (info.get("percent") or 0) >= 100:
            blocked.append(name)
    return blocked


def format_usage(usage: dict) -> str:
    now = datetime.now(timezone.utc)
    lines = []
    for name, info in usage.items():
        if not isinstance(info, dict):
            continue
        pct = int(info.get("percent") or 0)
        filled = min(20, pct // 5)
        bar = "#" * filled + "." * (20 - filled)
        status = info.get("status", "?")
        left = ""
        raw_reset = info.get("resetsAt")
        if raw_reset:
            try:
                dt = datetime.fromisoformat(str(raw_reset).replace("Z", "+00:00"))
                delta = dt - now
                if delta.total_seconds() > 0:
                    days, rem = divmod(int(delta.total_seconds()), 86400)
                    hours = rem // 3600
                    left = f"  resets in {days}d {hours}h"
            except ValueError:
                pass
        flag = "  <-- BLOCKING" if status == "rate-limited" or pct >= 100 else ""
        lines.append(f"  {name:<9} [{bar}] {pct:>3}%  {status}{left}{flag}")
    return "\n".join(lines) or "  (no usage data returned)"


def main() -> int:
    load_env_file()
    key = judge_api_key()
    if not key:
        print(f"error: none of {', '.join(API_KEY_ENVS)} found in env or .env", file=sys.stderr)
        return 2
    try:
        usage = fetch_usage(key)
    except httpx.HTTPError as err:
        print(f"error: could not fetch usage: {err}", file=sys.stderr)
        return 1

    print(f"OpenCode Zen Go quota ({USAGE_URL})\n")
    print(format_usage(usage))
    blocked = blocked_windows(usage)
    print()
    if blocked:
        print(f"BLOCKED: {', '.join(blocked)} limit reached — inference will 429.")
        return 1
    print("OK: quota available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
