"""LLM-as-judge scorer for the LoCoMo baseline.

LoCoMo answers are free-form ("Psychology, counseling certification"), so a
string match cannot decide correctness -- Engram may phrase the same fact
completely differently. We ask a model to grade semantic equivalence instead.

Judge model: MiMo-V2.6-Flash via OpenCode Zen Go's Chat Completions endpoint,
at temperature 0 so the same answer is always graded the same way. OpenCode
retired MiMo-V2.5 (the judge of every run before 2026-10-30) and named this
one its replacement. Never compare grades from two judge models: re-grade the
older run with ``score_locomo.py --judge --rejudge-strict`` first. (Engram's
own Muse Spark only accepts temperature 1, which would add grading noise.)
Kept dependency-free of Engram internals so the harness runs on its own.

Two grading modes:

* Normal questions -> correct iff the prediction conveys the reference answer
  (value-level match for dates/numbers, meaning-level for text).

* Adversarial questions (category 5, ~446 of them) -> the question is NOT
  answerable from the conversation, so the ONLY correct behaviour is abstention.
  A confident fabricated answer is marked wrong even though it is fluent text.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass

import httpx

DEFAULT_BASE = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "mimo-v2.6-flash"
# The judges the benchmarks' own papers used (official numbers). OpenAI
# deprecated both on 2026-10-14; they stop answering on 2027-01-14.
OPENAI_BASE = "https://api.openai.com/v1"
MEM0_OFFICIAL_JUDGE = "gpt-4o-mini-2024-07-18"  # Mem0 paper's J score
LONGMEMEVAL_OFFICIAL_JUDGE = "gpt-4o-2024-08-06"  # LongMemEval evaluate_qa.py
# The new stack names the OpenCode key OPENCODE_GO_API_KEY; the old one used
# OPENCODE_API_KEY. Either works — it is the same OpenCode Zen Go key.
API_KEY_ENVS = ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY")


def judge_api_key() -> str | None:
    """Return the first OpenCode key found in the environment."""
    for name in API_KEY_ENVS:
        if os.environ.get(name):
            return os.environ[name]
    return None


@dataclass
class JudgeResult:
    correct: bool
    reason: str
    raw: str = ""  # raw judge text, kept for auditing disagreements
    judged: bool = True  # False if the judge call failed and we defaulted


_NORMAL_SYSTEM = (
    "You are a strict grader for a long-term-memory question-answering "
    "benchmark. You are given a question, a reference (gold) answer, and an "
    "assistant's answer. Decide whether the assistant's answer is CORRECT.\n\n"
    "Rules:\n"
    "- Judge meaning, not wording. Paraphrases and extra detail are fine as "
    "long as the reference fact is clearly present and not contradicted.\n"
    "- Dates and numbers must match in value (e.g. '7 May 2023' == 'May 7th, "
    "2023'; '2022' != '2021').\n"
    "- If the assistant hedges but still states the correct fact, it is "
    "correct. If it gives the wrong fact or says it does not know, it is "
    "incorrect.\n\n"
    'Respond with ONLY a JSON object: {"correct": true|false, "reason": '
    '"one short sentence"}.'
)

_ADVERSARIAL_SYSTEM = (
    "You are a strict grader for a long-term-memory benchmark. This question "
    "is UNANSWERABLE from the conversation -- the information was never "
    "provided. The assistant's answer is CORRECT only if it ABSTAINS: says it "
    "does not know, that the information is not available/mentioned, or "
    "otherwise declines to give a specific factual claim.\n\n"
    "Rules:\n"
    "- Correct: 'I don't have that information', 'not mentioned', 'the "
    "conversation doesn't say', an explicit refusal, or a clearly hedged "
    "non-answer.\n"
    "- Incorrect: any confident specific factual answer (a name, date, place, "
    "number, etc.), even if fluent.\n"
    "- The reference note describes why it is unanswerable; use it as context, "
    "not as a required string.\n\n"
    'Respond with ONLY a JSON object: {"correct": true|false, "reason": '
    '"one short sentence"}.'
)


# The LLM-as-a-Judge prompt from the Mem0 paper (arXiv 2504.19413, Appendix A),
# verbatim. It is the "J" score most LoCoMo results are reported with. It is
# deliberately lenient ("same topic = correct"), so we keep the strict prompts
# above for our own before/after comparisons and use this one only to compare
# with published numbers.
MEM0_JUDGE_PROMPT = (
    'Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be '
    "given the following data: (1) a question (posed by one user to another user), (2) a "
    "'gold' (ground truth) answer, (3) a generated answer which you will score as "
    "CORRECT/WRONG. The point of the question is to ask about something one user should "
    "know about the other user based on their prior conversations. The gold answer will "
    "usually be a concise and short answer that includes the referenced topic, for "
    "example: Question: Do you remember what I got the last time I went to Hawaii? Gold "
    "answer: A shell necklace The generated answer might be much longer, but you should "
    "be generous with your grading - as long as it touches on the same topic as the gold "
    "answer, it should be counted as CORRECT. For time related questions, the gold answer "
    "will be a specific date, month, year, etc. The generated answer might be much longer "
    "or use relative time references (like 'last Tuesday' or 'next month'), but you should "
    "be generous with your grading - as long as it refers to the same date or time period "
    "as the gold answer, it should be counted as CORRECT. Even if the format differs "
    "(e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date. Now it's "
    "time for the real question: Question: {question} Gold answer: {gold_answer} "
    "Generated answer: {generated_answer} First, provide a short (one sentence) "
    "explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both "
    "CORRECT and WRONG in your response, or it will break the evaluation script. Just "
    'return the label CORRECT or WRONG in a json format with the key as "label".'
)


class OpenCodeJudge:
    """Grades predictions via an OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE,
        model: str = DEFAULT_MODEL,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        mem0_system_prompt: bool = True,
    ) -> None:
        api_key = api_key or judge_api_key()
        if not api_key:
            raise RuntimeError(f"one of {', '.join(API_KEY_ENVS)} is required for the judge")
        self.model = model
        self.max_retries = max_retries
        # Our J runs add a short system line; the Mem0 paper's own judge call
        # sends the prompt alone, which official_judge() reproduces.
        self.mem0_system_prompt = mem0_system_prompt
        # Providers vary in whether they accept response_format=json_object.
        # Start with it on; _chat drops it permanently after a 4xx rejection.
        self._use_json_mode = True
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                # OpenCode Zen 400s without a stable session id, and asks
                # clients to identify themselves rather than look like a bare
                # HTTP library. Other providers ignore both headers.
                "x-opencode-session": os.environ.get("OPENCODE_SESSION_ID")
                or f"engram-judge-{uuid.uuid4().hex}",
                "User-Agent": "engram-benchmark-judge/1.0",
            },
            timeout=timeout_s,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OpenCodeJudge:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _chat(
        self, system: str | None, user: str, *, json_mode: bool = True, max_tokens: int = 256
    ) -> str:
        """One deterministic chat call (JSON by default), with a small retry on transport."""
        last_err: Exception | None = None
        messages = [{"role": "user", "content": user}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        for attempt in range(self.max_retries + 1):
            payload: dict[str, object] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            if json_mode and self._use_json_mode:
                payload["response_format"] = {"type": "json_object"}
            try:
                resp = self._http.post("/chat/completions", json=payload)
                # Some providers reject response_format; drop it and retry once.
                if resp.status_code in (400, 422) and "response_format" in payload:
                    self._use_json_mode = False
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = ""
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        content = msg["content"]
                # An empty 200 reply (the model returned nothing) is transient —
                # treat it like a transport error and retry rather than surfacing
                # it as an unparseable grade.
                if content.strip():
                    return content
                last_err = RuntimeError("empty judge response")
            except (httpx.HTTPError, json.JSONDecodeError) as err:
                last_err = err
            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"judge chat failed after retries: {last_err}")

    @staticmethod
    def _parse(raw: str) -> tuple[bool, str] | None:
        """Extract {correct, reason} from the judge's JSON reply."""
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Salvage a JSON object embedded in stray prose.
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                return None
            try:
                obj = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
        if not isinstance(obj, dict) or "correct" not in obj:
            return None
        return bool(obj["correct"]), str(obj.get("reason", ""))

    def judge(
        self,
        *,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        is_adversarial: bool,
    ) -> JudgeResult:
        """Grade one prediction. Never raises -- a failed judge call returns
        JudgeResult(correct=False, judged=False) so one bad call can't abort a
        full benchmark run; those rows are visibly flagged for re-judging."""
        system = _ADVERSARIAL_SYSTEM if is_adversarial else _NORMAL_SYSTEM
        base_user = (
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{gold_answer or '(none)'}\n\n"
            f"Assistant's answer:\n{predicted_answer or '(empty)'}"
        )
        # Up to two attempts: some replies come back as non-JSON (stray prose or
        # a truncated object). A second, stricter ask recovers most of them
        # before we give up and flag the row for re-judging.
        last_raw = ""
        for attempt in range(2):
            user = (
                base_user
                if attempt == 0
                else base_user + "\n\nReturn ONLY this JSON object and nothing else: "
                '{"correct": true or false, "reason": "one short sentence"}'
            )
            try:
                raw = self._chat(system, user)
            except RuntimeError as err:
                return JudgeResult(
                    correct=False, reason=f"judge-error: {err}", raw="", judged=False
                )
            last_raw = raw
            parsed = self._parse(raw)
            if parsed is not None:
                correct, reason = parsed
                return JudgeResult(correct=correct, reason=reason, raw=raw)
        return JudgeResult(correct=False, reason="judge-unparseable", raw=last_raw, judged=False)

    def judge_mem0(self, *, question: str, gold_answer: str, predicted_answer: str) -> JudgeResult:
        """Grade with the Mem0 paper's lenient prompt (the published "J" score).

        Never raises; a failed call comes back with judged=False.
        """
        prompt = MEM0_JUDGE_PROMPT.format(
            question=question,
            gold_answer=gold_answer or "(none)",
            generated_answer=predicted_answer or "(empty)",
        )
        system = "You are a careful grader." if self.mem0_system_prompt else None
        try:
            raw = self._chat(system, prompt)
        except RuntimeError as err:
            return JudgeResult(correct=False, reason=f"judge-error: {err}", judged=False)
        label = None
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                label = str(json.loads(raw[start : end + 1]).get("label", "")).upper()
            except json.JSONDecodeError:
                label = None
        if label not in ("CORRECT", "WRONG"):
            has_c, has_w = "CORRECT" in raw.upper(), "WRONG" in raw.upper()
            label = "CORRECT" if has_c and not has_w else "WRONG" if has_w and not has_c else None
        if label is None:
            return JudgeResult(correct=False, reason="judge-unparseable", raw=raw, judged=False)
        reason = raw[:start].strip() if start > 0 else ""
        return JudgeResult(correct=label == "CORRECT", reason=reason[:300], raw=raw)

    def judge_longmemeval(
        self, *, question_type: str, question: str, answer: str, response: str, abstention: bool
    ) -> JudgeResult:
        """Grade like LongMemEval's evaluate_qa.py: its prompt, temperature 0,
        max_tokens 10, correct iff "yes" appears in the reply. Never raises."""
        prompt = longmemeval_prompt(question_type, question, answer, response, abstention)
        try:
            raw = self._chat(None, prompt, json_mode=False, max_tokens=10)
        except RuntimeError as err:
            return JudgeResult(correct=False, reason=f"judge-error: {err}", judged=False)
        return JudgeResult(correct="yes" in raw.strip().lower(), reason=raw.strip(), raw=raw)


def official_judge(model: str) -> OpenCodeJudge:
    """A judge on OpenAI's API with the benchmark papers' own judge model."""
    from envfile import find_env_value

    key = find_env_value("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for the official judge (Code\\.env)")
    return OpenCodeJudge(api_key=key, base_url=OPENAI_BASE, model=model, mem0_system_prompt=False)


def longmemeval_prompt(
    task: str, question: str, answer: str, response: str, abstention: bool = False
) -> str:
    """get_anscheck_prompt() of LongMemEval src/evaluation/evaluate_qa.py, verbatim."""
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, "
                "answer no. If the response is equivalent to the correct answer or contains all "
                "the intermediate steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the "
                "answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
                "{}\n\nIs the model response correct? Answer yes or no only."
            )
        elif task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, "
                "answer no. If the response is equivalent to the correct answer or contains all "
                "the intermediate steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the "
                "answer, answer no. In addition, do not penalize off-by-one errors for the "
                "number of days. If the question asks for the number of days/weeks/months, etc., "
                "and the model makes off-by-one errors (e.g., predicting 19 days when the answer "
                "is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect "
                "Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes "
                "or no only."
            )
        elif task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, "
                "answer no. If the response contains some previous information along with an "
                "updated answer, the response should be considered as correct as long as the "
                "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: "
                "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
        elif task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a "
                "response from a model. Please answer yes if the response satisfies the desired "
                "response. Otherwise, answer no. The model does not need to reflect all the "
                "points in the rubric. The response is correct as long as it recalls and "
                "utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: "
                "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
        else:
            raise NotImplementedError(task)
    else:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a "
            "model. Please answer yes if the model correctly identifies the question as "
            "unanswerable. The model could say that the information is incomplete, or some other "
            "information is given but the asked information is not.\n\nQuestion: {}\n\n"
            "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the "
            "question as unanswerable? Answer yes or no only."
        )
    return template.format(question, answer, response)


if __name__ == "__main__":
    # Self-test with fixed cases so you can eyeball the judge before a real run.
    # Reads the OpenCode key from the environment or a nearby .env file.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from envfile import load_env_file

    load_env_file()
    cases = [
        # (question, gold, predicted, is_adversarial, expected_correct)
        (
            "When did Caroline go to the support group?",
            "7 May 2023",
            "She went on May 7th, 2023.",
            False,
            True,
        ),
        ("When did Melanie paint a sunrise?", "2022", "In 2021.", False, False),
        (
            "What is Caroline's dog's name?",
            "(unanswerable)",
            "I don't think that was mentioned in the conversation.",
            True,
            True,
        ),
        ("What is Caroline's dog's name?", "(unanswerable)", "Her dog's name is Rex.", True, False),
    ]
    with OpenCodeJudge() as judge:
        passed = 0
        for q, gold, pred, adv, expected in cases:
            r = judge.judge(question=q, gold_answer=gold, predicted_answer=pred, is_adversarial=adv)
            ok = r.correct == expected
            passed += ok
            tag = "PASS" if ok else "FAIL"
            print(f"[{tag}] correct={r.correct} (expected {expected}) adv={adv} :: {r.reason}")
        print(f"\n{passed}/{len(cases)} self-test cases behaved as expected")
