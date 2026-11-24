# Memory systems on LoCoMo and LongMemEval: who Engram must be compared with

Snapshot of 24 November 2026. Numbers were read from the papers and pages
linked at the end; "self-reported" means the system's own authors or vendor
measured it.

## 1. Read this before comparing any number

- **The answer model decides a lot.** The same system scores very differently
  with a stronger model: EverMemOS gets 86.8 on LoCoMo with gpt-4o-mini and
  93.1 with gpt-4.1-mini. Many 2026 results use gpt-4.1-mini, GPT-5 or
  Gemini-3. Engram's runs use `muse-spark-1.3-contributor`.
- **Who ran the test matters even more.** Mem0 on LoCoMo: 66.9 (its own
  paper, 2025), 61.0–64.6 (MemOS and EverMemOS papers), 77.7 (an independent
  harness), 92.5 (Mem0's 2026 hosted platform, self-reported). Zep: 66.0
  (run by Mem0), 75.1 (run by Zep), 63.8 (independent harness).
- **Same metric name, different test.** LoCoMo "J" (LLM-judge accuracy) is
  normally on the 1,540 non-adversarial questions with Mem0's judge prompt;
  LoCoMo "F1" is the original word-overlap score. LongMemEval is normally the
  500-question **S** version graded by gpt-4o; some vendors report subsets or
  retrieval recall instead.
- **Fair rule for the paper:** only compare rows with the same answer model and
  judge, or run the baselines ourselves under our harness.

## 2. LoCoMo

**Table 1 — LoCoMo J (%), answer model gpt-4o-mini (the Mem0 protocol, 2025).**

| System | J | Who measured | Source |
|---|---|---|---|
| Full context (whole conversation, no memory system) | 72.9 | Mem0 paper | [Mem0] |
| MemOS (v1031) | 75.8 | self-reported | [MemOS] |
| Memobase | 75.8 | self-reported | [Hindsight] table |
| Zep | 75.1 | self-reported (Zep's correction of Mem0's run) | [Zep blog] |
| Letta (agent + file tools over the raw conversation) | 74.0 | self-reported | [Letta] |
| Mem0 graph / Mem0 | 68.4 / 66.9 | self-reported | [Mem0] |
| Zep (run by Mem0) | 66.0 | Mem0 paper | [Mem0] |
| Best plain RAG | 61.0 | Mem0 paper | [Mem0] |
| LangMem / OpenAI memory / A-Mem | 58.1 / 52.9 / 48.4 | Mem0 paper | [Mem0] |
| EverMemOS (vs Zep 81.1, MemOS 75.9, Mem0 61.0 in the same table) | 86.8 | self-reported | [EverMemOS] |

**Table 2 — LoCoMo J (%), stronger answer models (2026).**

| System | J | Setup | Source |
|---|---|---|---|
| EverMemOS (vs Zep 85.2, MemOS 80.8, Mem0 64.2) | 93.1 | gpt-4.1-mini, self-reported | [EverMemOS] |
| Hindsight | 89.6 / 85.7 | Gemini-3 / GPT-OSS-120B, self-reported | [Hindsight] |
| Mem0 hosted platform (2026) | 92.5 | gpt-4o answers and judge, top-200 retrieval, self-reported | [Mem0 2026] |
| Independent harness, same setup for all (answer gpt-4.1-mini, judge gpt-4o-mini) | cachly 90.7, ThreadAtlas 88.9, **MemOS 88.8**, Maasv 87.7, Cognee 83.5, EverOS 82.8, Hindsight 82.0, **Mem0 77.7**, Letta 77.1, MemMachine 73.9, Supermemory 73.5, **Zep 63.8** | measured by MemTensor | [OmniMemEval] |

**Official LoCoMo F1** (the original paper's metric) is rarely reported by
memory systems. The LoCoMo paper itself (50 conversations, not the 10 public
ones) gives overall F1: humans 87.9, best RAG (GPT-3.5) 41.4, GPT-4-turbo
32.1. Papers using the Mem0 protocol report F1 on categories 1–4 of about
41–50 (MemOS 45.3, Mem0 43.5, Zep 41.2 in [MemOS]; Memory-R1 45.0 in
[Memory-R1]).

## 3. LongMemEval-S (500 questions, accuracy %, official gpt-4o grading unless noted)

| System | Accuracy | Setup | Source |
|---|---|---|---|
| Full context, GPT-4o / gpt-4o-mini | 60.6 / 55.4 | no memory system; oracle (only the answer chats) 87.0 | [LongMemEval], [Zep] |
| Zep | 71.2 / 63.8 | gpt-4o / gpt-4o-mini, self-reported | [Zep] |
| Mem0 / MemOS | 66.4 / 77.8 | gpt-4o-mini, MemOS leaderboard | [MemOS], [EverMemOS] |
| EverMemOS | 83.0 | self-reported | [EverMemOS] |
| Nemori | 74.6 / 64.2 | gpt-4.1-mini / gpt-4o-mini; full context 65.6 / 55.0 | [Nemori] |
| Supermemory | 81.6 / 85.2 | gpt-4o / Gemini-3, self-reported | [Hindsight] table |
| Hindsight | 91.4 / 83.6 | Gemini-3 / GPT-OSS-20B, self-reported | [Hindsight] |
| Independent harness (gpt-4.1-mini, judge gpt-4o-mini) | Maasv 91.2, cachly 89.6, **MemOS 89.2**, LCM-X 87.0, ThreadAtlas 87.0, EverOS 80.4, **Zep 79.8**, Letta 77.7, Hindsight 72.2, Supermemory 66.1, MemMachine 63.6, **Mem0 56.0** | measured by MemTensor | [OmniMemEval] |
| 2026 vendor claims | Mem0 94.4 (another vendor measured 73.8), Synap 95.6, MemMachine 93.0 | self-reported | [Mem0 2026], [Maximem] |

## 4. Where Engram stands and what it has to beat

**Engram today** (LoCoMo in the official format, run
`locomo-official2-combined`; answer model `muse-spark-1.3-contributor`):

| | Overall | Multi-hop | Temporal | Open-domain | Single-hop |
|---|---|---|---|---|---|
| J, gpt-4o-mini judge (1,540 questions) | **67.1** | 66.7 | 76.0 | 40.6 | 66.8 |
| F1, categories 1–4 | 44.2 | 31.5 | 54.9 | 17.4 | 47.5 |
| Official F1, all 1,986 (adversarial 94.0) | **55.4** | | | | |

LongMemEval: only a 6-question pilot so far (4/6), so no number to report yet.

**Reading the tables:**
- On the 2025 protocol, Engram's J (67.1) equals **Mem0 (66.9)** and **Zep as
  run by Mem0 (66.0)**. It is still **below the full-context baseline (72.9)**,
  Letta (74.0), Zep's own run (75.1) and MemOS (75.8).
- The 2026 leaders reach 85–93 on LoCoMo and 83–91 on LongMemEval-S, mostly
  with stronger answer models and often self-measured. On one shared harness,
  the best systems are around 88–91 on both benchmarks.
- Engram's F1 on categories 1–4 (44.2) is in the same range as MemOS (45.3)
  and Mem0 (43.5).
- Engram's weakest areas are **open-domain** (40.6 J) and, under our stricter
  internal judge, **multi-hop** (31.6 strict, though 66.7 J).

**Targets, from easiest to hardest:**
1. **Beat full context with our own answer model.** A memory system must beat
   simply giving the model the whole conversation (72.9 with gpt-4o-mini).
   We have not run this baseline with `muse-spark` yet; it is cheap on LoCoMo
   (about 26k tokens per conversation).
2. **Match the 2025 state of the art:** J ≥ 75–76 on LoCoMo (Zep, MemOS,
   Memobase) and ≥ 71–78 on LongMemEval-S (Zep, MemOS).
3. **2026 state of the art:** about 85–93 on LoCoMo and 83–91 on LongMemEval-S.
   Fair only with the same answer model; on the shared harness this means ≥ 88.

**Recommended for the paper:**
- Run a full-context baseline with our answer model on LoCoMo and on our
  LongMemEval sample.
- Re-run two or three open-source baselines (Mem0 OSS, Zep/Graphiti, MemOS or
  EverMemOS) under our harness and models; or adopt the shared harness setup
  (gpt-4.1-mini answers, gpt-4o-mini judge) so our numbers sit in a published
  table.
- Report J (gpt-4o-mini judge), official F1 and per-category scores, with
  adversarial separate, and name the answer and judge models in every table.

## Sources

- [LoCoMo] Maharana et al., ACL 2024 — <https://arxiv.org/abs/2402.17753>, code <https://github.com/snap-research/locomo>
- [LongMemEval] Wu et al., ICLR 2025 — <https://arxiv.org/abs/2410.10813>, code <https://github.com/xiaowu0162/LongMemEval>
- [Mem0] Chhikara et al., 2025 — <https://arxiv.org/abs/2504.19413>, code <https://github.com/mem0ai/mem0>
- [Mem0 2026] Mem0 benchmark page — <https://mem0.ai/library/agent-memory/ai-memory-benchmarks-in-2026>
- [Zep] Rasmussen et al., 2025 — <https://arxiv.org/abs/2501.13956>, code <https://github.com/getzep/graphiti>
- [Zep blog] "Is Mem0 really SOTA?" — <https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/>
- [Letta] "Benchmarking AI agent memory" — <https://www.letta.com/blog/benchmarking-ai-agent-memory>
- [MemOS] MemTensor, 2025 — <https://arxiv.org/abs/2507.03724>, code <https://github.com/MemTensor/MemOS>
- [EverMemOS] EverMind, 2026 — <https://arxiv.org/abs/2601.02163>, code <https://github.com/EverMind-AI/EverMemOS>
- [Hindsight] Vectorize, 2025 — <https://arxiv.org/abs/2512.12818>, code <https://github.com/vectorize-io/hindsight>
- [Nemori] 2025 — <https://arxiv.org/abs/2508.03341>, code <https://github.com/nemori-ai/nemori>
- [Memory-R1] 2025 — <https://arxiv.org/abs/2508.19828>; A-Mem <https://arxiv.org/abs/2502.12110>; MemoryOS <https://arxiv.org/abs/2506.06326>
- [OmniMemEval] independent harness by MemTensor — <https://github.com/MemTensor/OmniMemEval/blob/main/docs/user_memory/results.md>
- [Maximem] "State of AI memory 2026: claimed vs observed" — <https://www.maximem.ai/blog/state-of-ai-memory-2026-claimed-vs-observed>
