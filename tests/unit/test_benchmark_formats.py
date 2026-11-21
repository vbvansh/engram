"""The benchmark harness follows each benchmark's official QA and scoring format."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from judge import longmemeval_prompt  # noqa: E402
from run_locomo import (  # noqa: E402
    OFFICIAL_NOT_MENTIONED,
    OFFICIAL_TEMPORAL_HINT,
    official_instructions,
)
from run_longmemeval import lme_timestamp, session_pairs  # noqa: E402
from score_locomo import locomo_f1  # noqa: E402
from score_longmemeval import metrics  # noqa: E402


def _probe(category: int, answer: str = "a trophy") -> SimpleNamespace:
    return SimpleNamespace(category=category, answer=answer)


def test_locomo_official_format_adds_the_date_hint_and_the_adversarial_options():
    assert official_instructions(_probe(2), "conv-26:3") == OFFICIAL_TEMPORAL_HINT
    assert official_instructions(_probe(4), "conv-26:4") is None

    options = official_instructions(_probe(5), "conv-26:9")
    assert options.startswith("Select the correct answer: (a) ")
    assert OFFICIAL_NOT_MENTIONED in options and "a trophy" in options
    # The order is random per question, as in the official code, but stable.
    assert official_instructions(_probe(5), "conv-26:9") == options
    orders = {official_instructions(_probe(5), f"conv-26:{i}").index("a trophy") for i in range(20)}
    assert len(orders) == 2


def test_locomo_official_f1_rewards_short_answers_and_the_not_mentioned_choice():
    assert locomo_f1("UK", "UK", 4) == 1.0
    assert locomo_f1("Tim was visiting the UK in the second week of November.", "UK", 4) < 0.2
    assert locomo_f1("Not mentioned in the conversation", "a trophy", 5) == 1.0
    assert locomo_f1("No information available.", "a trophy", 5) == 1.0
    assert locomo_f1("I don't have that information.", "a trophy", 5) == 0.0


def test_longmemeval_history_keeps_roles_order_and_dates():
    turns = [
        {"role": "assistant", "content": "Welcome back!"},
        {"role": "user", "content": "I moved to Denver."},
        {"role": "assistant", "content": "Nice!"},
        {"role": "user", "content": "Bye."},
    ]
    assert session_pairs(turns) == [
        ("", "Welcome back!"), ("I moved to Denver.", "Nice!"), ("Bye.", "")
    ]
    assert lme_timestamp("2023/05/20 (Sat) 02:21") == "2023-05-20T02:21:00"


def test_longmemeval_metrics_match_print_qa_metrics():
    def row(qtype, label, abstention=False):
        return {"question_type": qtype, "abstention": abstention,
                "autoeval_label": {"model": "gpt-4o-2024-08-06", "label": label}}

    m = metrics([
        row("multi-session", True), row("multi-session", False),
        row("knowledge-update", True, abstention=True),
    ])
    assert m["overall_accuracy"] == 2 / 3
    assert m["task_averaged_accuracy"] == (0.5 + 1.0) / 2
    assert m["abstention_accuracy"] == 1.0 and m["abstention_n"] == 1
    assert m["by_type"]["temporal-reasoning"] == {"accuracy": None, "n": 0}


def test_longmemeval_judge_prompt_depends_on_type_and_abstention():
    temporal = longmemeval_prompt("temporal-reasoning", "Q", "A", "R")
    assert "do not penalize off-by-one errors" in temporal
    unanswerable = longmemeval_prompt("multi-session", "Q", "A", "R", abstention=True)
    assert "correctly identifies the question as unanswerable" in unanswerable
