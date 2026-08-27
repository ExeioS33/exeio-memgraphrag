# Evaluation Harness

> **Status: implemented.** `memgraphrag/evaluation/` ships the metrics, the
> dataset loaders, the judge prompt, the variance-aware runner and the
> golden-set check. `scripts/evaluate.py` measures quality, `scripts/bench.py`
> measures performance. This supersedes the "no harness ships in this
> repository" note in [Reproduce.md](Reproduce.md) for everything below;
> load generation against the HTTP surface is still out of scope here.

Every number this harness prints is defined **here**, not in the paper. That is
not a stylistic choice: the definitions do not exist upstream, and the sections
below say exactly where each one was frozen and what was rejected.

## Why the definitions had to be frozen

[Wu et al., arXiv:2606.00610](https://arxiv.org/abs/2606.00610) evaluates with
four columns — Str-Acc, LLM-Acc, Context Relevance, Evidence Recall — and none of
them is reproducible from the paper or from the released research code:

| Metric | What the paper gives | What is missing |
|--------|----------------------|-----------------|
| Str-Acc | "whether the gold answer is included in the generated answer after normalizing them to lowercase words" | Substring or token match? Which normalization? What stops a long answer from containing the gold span by accident? |
| LLM-Acc | A column of numbers | No judge prompt, no judge model, no parsing rule. Absent from the paper **and** from the published code |
| Context Relevance | A reference to GraphRAG-Bench | No formula, no unit of comparison |
| Evidence Recall | A reference to GraphRAG-Bench | Same |

The research repository (`MemGraphRAG/code/src/evaluation/`) implements
`QAExactMatch`, `QAF1Score` and `RetrievalRecall` — three metrics that appear in
**no** column of the paper's tables. So the published code does not compute the
published numbers.

EXEIO's answer: implement all seven (the four paper metrics plus the EM / F1 /
Recall@k floor ported from the research code), freeze each definition in code,
and document the choice next to the alternative it beat.

## Normalization (`evaluation/normalization.py`)

Applied in this order, and shared by every answer-side metric:

1. Fold typographic characters onto ASCII (`–` → `-`, `’` → `'`, `…` → `...`), then NFKC.
2. Strip Markdown decoration (`**`, `__`, `` ` ``, headings, list bullets, link syntax) and inline citation markers (`[1]`, `【1】`).
3. Lowercase.
4. Replace punctuation with **spaces**.
5. Drop the articles `a` / `an` / `the`.
6. Collapse whitespace.

Two departures from the MRQA recipe used by the research code
(`MemGraphRAG/code/src/utils/eval_utils.py`):

- **Punctuation becomes a separator, not a deletion.** MRQA deletes it, which
  welds tokens together: `U.S.-based` normalizes to `usbased` and can never match
  `us based`.
- **Markdown is stripped.** The research script scored short extractive spans;
  this server returns Markdown prose, and `**Chris Evans**` must not be a miss.

Document identities use a *different* normalizer (`normalize_doc_key`): articles
are **kept**, because "The Newcomers" and "Newcomers" are two distinct Wikipedia
titles, and merging them would inflate retrieval scores.

## Str-Acc — frozen definition

> Str-Acc = 1 when the normalized token sequence of any gold answer occurs as a
> **contiguous subsequence** of the first **200** normalized tokens of the
> prediction.

| Choice | Decision | Why |
|--------|----------|-----|
| Substring vs. tokens | **Contiguous token containment** | A raw substring test fires on `art` inside `started` and `US` inside `USSR`. Those accidental hits land exactly on the short gold answers (dates, yes/no, single names) that dominate HotpotQA and 2Wiki |
| Contiguity | **Required** | Bag-of-tokens containment accepts "York is new to the list" for the gold answer "New York" |
| Length bound | **First 200 tokens** (`STR_ACC_WINDOW_TOKENS`) | Inclusion is monotone in answer length: with no bound, a long enough answer eventually contains almost any short gold string, so an unbounded Str-Acc rewards verbosity. 200 tokens ≈ a direct answer plus one paragraph of justification |
| Multiple gold answers | **Max over golds** | Same aggregation as the research EM/F1 code |
| Empty gold answer | **Never a hit** | Otherwise every prediction, including the empty one, scores a free point |

Because the bound is a judgement call, every run also reports the two counters
that let a reader audit it:

- `StrAccVerboseHitRate` — share of **hits** whose prediction is more than 10×
  longer than the gold answer.
- `StrAccTruncatedRate` — share of predictions long enough for the 200-token
  window to be load-bearing.

**A Str-Acc figure quoted without those two numbers is not reportable inside
EXEIO.** They are what separates an accuracy gain from a verbosity gain.

## LLM-Acc — the judge prompt lives in the code

`evaluation/judge.py` holds the system prompt, the user template and a version
string, `LLM_ACC_PROMPT_VERSION = "exeio-llm-acc-v1"`, written into every run
report. Reconstructed from the one published judge template in this workspace,
`LightRAG/reproduce/batch_eval.py` (role line, explicit criteria, mandated JSON
object), with two deliberate departures:

- LightRAG's judge compares **two** answers and picks a winner: that yields a
  preference, not an accuracy. This judge grades one answer against the gold
  answer and returns a binary verdict.
- The verdict is a JSON object `{"verdict": "correct"|"incorrect", "reason": …}`,
  parsed with the repository's tolerant extractor.

Operating rules:

| Rule | Value | Why |
|------|-------|-----|
| Temperature | `0.0` | Judge noise would be indistinguishable from engine noise in the variance figures |
| Seed | `7`, sent when the endpoint accepts it | Same |
| Judge model | `--judge-model`, defaulting to `LLM_MODEL` | **Prefer a different model from the one under test.** Grading with the model being graded is how a benchmark flatters itself |
| Unreadable reply | `verdict="unparsed"`, counted as incorrect **and** reported in `LLMAccUnparsedRate` | A judge that stopped returning JSON otherwise looks exactly like an engine that got worse |
| Judge exception | Recorded as `verdict="error"`, the run continues | One dead call must not void a campaign |
| Prompt edit | **Bump the version string** | Two scores from two prompt versions are not comparable |

Untrusted text (question, gold, candidate) is fenced in `<<<NAME>>>` markers and
defanged, matching the untrusted-input handling in `memgraphrag/prompts/`.

## Context Relevance and Evidence Recall — frozen definitions

Both compare **document identities** (Wikipedia titles, or the source label for
the medical set), never passage text — a text comparison would make the score
depend on the chunker.

| Metric | Definition | Reading |
|--------|-----------|---------|
| **Evidence Recall** | `|gold ∩ retrieved| / |gold|` over the **whole** retrieved context, no fixed k | "Did the answer have a chance of being grounded?" |
| **Context Relevance** | `|gold ∩ retrieved| / |retrieved|`, retrieved de-duplicated by identity | "How much of the context window was signal?" |
| **Recall@k** | Gold documents found in the top-k, k ∈ {1, 5, 10, 20} | "Is the ranker good?" |

Evidence Recall and Context Relevance are precision/recall duals and are always
reported together: retrieving everything wins the first and destroys the second.
Raising `TOP_K` moves Evidence Recall without moving Recall@5 — that is the
point of reporting both.

Edge cases, frozen: a question with no gold labels scores 0.0 (never dropped,
which would let an unlabelled slice raise the mean); an empty retrieval scores
0.0 on Context Relevance (an empty context is a failure, not perfect precision).

## Metric keys emitted by a run

These are the exact keys in a run report, a golden set and a comparison:

| Key | Meaning |
|-----|---------|
| `StrAcc` | Str-Acc as frozen above |
| `StrAccVerboseHitRate` | Share of hits whose prediction is >10x the gold length |
| `StrAccTruncatedRate` | Share of predictions cut by the 200-token window |
| `LLMAcc` | Judge verdict accuracy (unparsed counted as incorrect) |
| `LLMAccUnparsedRate` | Share of judge replies that could not be parsed |
| `EvidenceRecall` | Gold evidence found in the whole retrieved context |
| `ContextRelevance` | Share of the retrieved context that is gold evidence |
| `ExactMatch` | Whole-answer equality after normalization (MRQA floor) |
| `F1` | Bag-of-tokens F1 (MRQA floor) |
| `Recall@1`, `Recall@5`, `Recall@10`, `Recall@20` | Gold documents inside the top-k |

## Datasets (`evaluation/datasets.py`)

Four datasets, 1000 questions each, in the research checkout — referenced by
nothing in this repository before this harness existed.

| Name | Questions file | Shape | Corpus | Gold documents from |
|------|----------------|-------|--------|---------------------|
| `hotpotqa` | `hotpotqa/hotpotqa.json` | flat list | `hotpotqa_corpus.json` | `supporting_facts[][0]` titles |
| `2wikimultihopqa` | `2wikimutlhopqa/2wikimultihopqa.json` | flat list | `2wikimultihopqa_corpus.json` | `supporting_facts[][0]` titles |
| `musique` | `musique/musique.json` | `{source, questions: {type: […]}}` | `musique_corpus.json` | `paragraphs[].title` where `is_supporting` |
| `medical` | `medical/question.json` | `{source, questions: {type1…type4}}` | `medical.txt` only | the record's `source` field |

Notes that cost time if unknown:

- The research directory really is spelled `2wikimutlhopqa`. The loader accepts
  `2wikimultihopqa` (and `2wiki`) and finds the misspelled directory.
- MuSiQue gold answers include `answer_aliases`.
- Supporting facts list *sentences*; several may point at one document. Gold
  documents are de-duplicated, so a two-sentence-one-document question is not
  scored as needing two documents.
- The medical set ships no corpus JSON — only `medical.txt`, returned as one
  document. The caller's chunker decides how to split it; the loader does not
  invent a segmentation the dataset never specified.
- Shape detection is by content (does the record carry a `questions` mapping),
  not by dataset name, so an upstream reshuffle fails loudly instead of loading
  wrongly.

The root is `--dataset-root`, else `$MEMGRAPHRAG_DATASET_ROOT`, else the sibling
checkout `../MemGraphRAG/dataset`. Missing files raise `DatasetUnavailableError`
naming the path *and* the override — a developer without the 58 MB research tree
gets one sentence, not a `FileNotFoundError` from inside a metric.

## Variance protocol (why `--runs` exists)

The paper publishes no run count, no standard deviation and no confidence
interval, while claiming gains as small as **+2.10 absolute points**. There is
therefore no way to tell, from the paper alone, whether 58.9 against 59.25 is a
regression or noise.

EXEIO protocol:

1. Run `--runs N` (N ≥ 3; 5 for a golden set) on a **fixed** question set. Use
   `--limit` (first N, deterministic) or `--sample N --seed S` (seeded subset,
   with the seed recorded in the report) — never an unseeded subset, which would
   make the standard deviation measure the sampler rather than the engine.
2. The runner reports mean, sample standard deviation (n−1), min, max and a 95%
   confidence half-width per metric. A single run reports `stdev = 0.0`; read
   that as **unmeasured**, not stable.
3. Runs execute sequentially. Overlapping them would have them contend for the
   same LLM endpoint, and the latency figures would measure the campaign's own load.
4. Derive the non-regression threshold from **our** variance, not from a
   hand-picked constant.

## Golden sets and non-regression

A golden set (`format: "memgraphrag-golden-set"`) freezes both halves of a
reference: the question set (id, question, gold answers, gold documents) and the
metric distribution (mean, stdev, runs, min, max), plus the metadata that makes a
score comparable (mode, top_k, models, judge prompt version).

```bash
# 1. Build the reference from a multi-run campaign.
uv run python scripts/evaluate.py --dataset hotpotqa --limit 50 --runs 5 \
    --judge --write-golden data/eval/hotpotqa50.golden.json

# 2. Later, check a change against it.
uv run python scripts/evaluate.py --dataset hotpotqa --limit 50 --runs 3 \
    --judge --compare data/eval/hotpotqa50.golden.json --tolerance 2
```

Comparison rules:

- Tolerance is in **standard deviations** (`--tolerance`, default 2.0).
- σ is the **larger** of the reference and current standard deviations: the
  noisier of the two measurements decides what "the same result" means.
- Audit counters (`StrAccVerboseHitRate`, `StrAccTruncatedRate`,
  `LLMAccUnparsedRate`) are scored lower-is-better, so a tripled unparsed-judge
  rate is flagged even though the number went up.
- If neither side measured variance (a golden set built from one run), the check
  falls back to a crude 1-point absolute floor and labels the row
  `absolute-floor (no variance measured)`. Do not read that green as evidence.
- Exit code 1 on any regression, so CI can gate on it.

## Performance harness (`scripts/bench.py`)

| Measured | How |
|----------|-----|
| Retrieval latency p50 / p95 / p99 | Wall time of `MemGraphRAG.aretrieve` per question, nearest-rank percentiles (never interpolated: an interpolated p95 reports a latency no request experienced) |
| Throughput | Completed queries per wall second at fixed `--concurrency`; errors count in the denominator, because they consumed time |
| LLM calls and tokens per query | The completion function is wrapped by `CallMeter`, so every engine path is counted; index-time traffic is excluded from the query figures |
| Full QA latency | `--with-answer` times retrieval + generation instead of retrieval alone |

`--warmup` calls are executed and discarded: the first query of a process pays
for lazy storage loads and PPR graph construction, and letting that land in the
sample turns p95 into a measurement of start-up.

Two honest gaps:

- **Tokens are estimated locally** with the project tokenizer. The
  OpenAI-compatible binding in `memgraphrag/llm/` does not surface
  `response.usage`, so billed tokens must come from the provider. The estimate is
  still the right tool for comparing two configurations against one endpoint.
- **TTFB is deliberately absent.** `POST /query/stream` awaits the complete
  answer before its first SSE frame, so TTFB equals total latency by
  construction. Do not add the number until streaming is real.

The paper's **0.061 s per retrieval** is printed next to our p50 for contrast
only: their hardware, corpus size and embedding endpoint are all unstated, so a
difference is not by itself evidence of anything.

## Replication target: the paper's own arithmetic is wrong

Before any campaign chases a gap, note what the published tables actually say.

- **Table 1, MemGraphRAG row.** The mean of the eight published columns is
  **59.63**, but the printed `Avg.` is **59.25**.
- 59.25 is reproducible only with **G-Novel = 54.41** — the value printed in
  Table 3 — instead of the **57.41** printed in Table 1. The arithmetic closes
  exactly: `59.63 − (57.41 − 54.41) / 8 = 59.255`.
- The entire Δ column of Table 1 and the 59.68 average of Table 3 are coherent
  only with 54.41. So the outlier is the Table 1 G-Novel cell, not the average.

**Consequence for EXEIO campaigns: the replication target on G-Novel is 54.41,
not 57.41.** Chasing a 3-point gap that exists only as a typo would burn a
campaign and could get a correct implementation "fixed" until it reproduced a
misprint.

Two further corrections to the paper's SOTA framing, worth recording so that no
internal deck repeats them:

- **G-Novel: MemGraphRAG is not the best system.** HippoRAG2 reports 56.48
  against MemGraphRAG's corrected 54.41.
- **MuSiQue LLM-Acc: MemGraphRAG is not the best system.** 37.90 against 38.30.

Finally, none of the paper's numbers are reproducible in this repository as-is
anyway: the igraph PPR engine here is an adaptation, not a paper-exact
implementation (see [AGENTS.md](../AGENTS.md), "Divergences from the paper").
Treat the published figures as an order of magnitude, and treat **our** golden
sets as the regression baseline.

## Commands

```bash
# What is available?
uv run python scripts/evaluate.py --list-datasets

# Quality: 50 questions, 3 runs, judged, JSON report.
uv run python scripts/evaluate.py --dataset hotpotqa --limit 50 --runs 3 \
    --judge --output data/eval/hotpotqa.json

# Reuse an already indexed working directory (indexing is the expensive part).
uv run python scripts/evaluate.py --dataset hotpotqa --no-ingest \
    --working-dir data/eval_storage/hotpotqa --limit 50 --runs 3

# Performance: retrieval only, then the full QA path.
uv run python scripts/bench.py --dataset hotpotqa --corpus-limit 200 --queries 50
uv run python scripts/bench.py --dataset hotpotqa --no-ingest \
    --working-dir data/eval_storage/hotpotqa --queries 100 --concurrency 4 --with-answer
```
