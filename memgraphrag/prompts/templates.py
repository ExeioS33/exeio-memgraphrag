"""Prompt string templates for MemGraphRAG OpenIE, ontology, conflicts, and QA.

Provenance: simplified ports of MemGraphRAG research prompts under
``MemGraphRAG/code/src/prompts/`` (``templates/ner.py``,
``templates/triple_extraction.py``, ``templates/rag_qa_musique.py``,
``prompt.py`` ontology/conflict entries, and ``linking.py`` instructions).
Placeholders use ``str.Template`` / ``$name`` style.
"""

from __future__ import annotations

from string import Template

# ---------------------------------------------------------------------------
# Linking instructions (asymmetric embedding prefixes)
# ---------------------------------------------------------------------------

QUERY_TO_FACT = (
    "Given a question, retrieve relevant triplet facts that matches this question."
)
QUERY_TO_PASSAGE = (
    "Given a question, retrieve relevant documents that best answer the question."
)

LINKING_INSTRUCTIONS = {
    "query_to_fact": QUERY_TO_FACT,
    "query_to_passage": QUERY_TO_PASSAGE,
    "ner_to_node": (
        "Given a phrase, retrieve synonymous or relevant phrases that best match this phrase."
    ),
    "query_to_node": (
        "Given a question, retrieve relevant phrases that are mentioned in this question."
    ),
    "query_to_sentence": (
        "Given a question, retrieve relevant sentences that best answer the question."
    ),
}


def get_query_instruction(linking_method: str) -> str:
    """Return the embedding instruction for a linking method."""
    return LINKING_INSTRUCTIONS.get(linking_method, QUERY_TO_PASSAGE)


# ---------------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------------

NER_SYSTEM = """Your task is to extract named entities from the given paragraph.
Respond with a JSON object containing a list of entities under the key "named_entities".
"""

NER_USER_TEMPLATE = Template(
    """Extract named entities from the paragraph below.
Respond ONLY with JSON of the form {"named_entities": ["...", "..."]}.

Paragraph:
$passage
"""
)

# ---------------------------------------------------------------------------
# Triple extraction (NER-conditioned)
# ---------------------------------------------------------------------------

TRIPLE_EXTRACTION_SYSTEM = """Your task is to construct an RDF (Resource Description Framework) graph from the given passage and named entity list.
Respond with a JSON object containing a list of triples under the key "triples".

Requirements:
- Each triple is [subject, predicate, object] (three strings).
- Prefer triples that involve the provided named entities.
- Resolve pronouns to specific names.
- Output valid JSON only.
"""

TRIPLE_EXTRACTION_USER_TEMPLATE = Template(
    """Convert the paragraph into a JSON dict with a named entity list and a triple list.

Paragraph:
```
$passage
```

$named_entity_json

Respond ONLY with JSON of the form:
{"triples": [["head", "relation", "tail"], ...]}
"""
)

# ---------------------------------------------------------------------------
# Ontology extraction
# ---------------------------------------------------------------------------

ONTOLOGY_EXTRACTION_SYSTEM = """You are an ontology converter. Given a passage and a list of factual triples (head, relation, tail),
output ontology-level triples by replacing the head and tail entities with their most appropriate entity TYPE
while keeping the relation unchanged.

Requirements:
- Infer types from the passage context; be specific but concise (one label).
- Do NOT invent new relations; keep relation text exactly as provided.
- Preserve one-to-one alignment: each input triple must have exactly one ontology triple.
- If a type is unclear, return "Unknown".
- Output MUST be valid JSON.
"""

ONTOLOGY_EXTRACTION_USER_TEMPLATE = Template(
    """Convert the following triples to ontology-level triples by replacing head and tail with their entity types. Keep relation unchanged.

Passage:
$passage

Triples:
$triples

Return JSON with structure:
{
  "ontology_triples": [
    {
      "triple": ["head", "relation", "tail"],
      "ontology": ["head_type", "relation", "tail_type"]
    }
  ]
}
"""
)

# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

CONFLICT_DETECTION_SYSTEM = """You are an expert knowledge graph fact checker.

Given ONE target triple and multiple related triples, detect whether the target conflicts with each related triple.
Be conservative: avoid false positives.

Conflict categories:
1) mutual: truly mutually exclusive facts.
2) temporal: conflict depends on time overlap.
3) granularity: different specificity level (often compatible, not hard conflict).

Important anti-false-positive rules:
- Exact duplicates are NOT conflicts (mark as "duplicate").
- One-to-many predicates are usually NOT mutual conflicts.
- When uncertain, prefer "no hard conflict".
Output MUST be valid JSON only.
"""

CONFLICT_DETECTION_USER_TEMPLATE = Template(
    """Analyze the following triples for conflicts.

Target Triple:
$target_triple

Related Triples:
$related_triples

Output a JSON object with:
{
  "has_conflict": true/false,
  "conflicts": [
    {
      "triple1": ["head", "relation", "tail"],
      "triple2": ["head", "relation", "tail"],
      "conflict_type": "mutual|temporal|granularity|duplicate|none|uncertain",
      "is_hard_conflict": true/false,
      "needs_resolution": true/false,
      "conflict_reason": "brief explanation"
    }
  ],
  "conflicting_triple_ids": ["id1", "id2", ...]
}
"""
)

# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------

CONFLICT_RESOLUTION_SYSTEM = """You are an expert knowledge graph curator. Given conflicting triples and their source passages,
resolve the conflicts and produce corrected triples.

Strategies:
1. Mutual: keep the correct triple, discard incorrect ones.
2. Temporal: add time context to the relation when possible.
3. Granularity: keep compatible facts; clarify scope in the relation when needed.

Output MUST be valid JSON.
"""

CONFLICT_RESOLUTION_USER_TEMPLATE = Template(
    """Resolve the following conflicting triples using their source passages.

Conflicting Triples and Their Sources:
$conflicting_triples_with_sources

Output JSON:
{
  "resolved_triples": [
    {
      "original_triple": ["head", "relation", "tail"],
      "triple_id": "fact_id",
      "conflict_type": "mutual|temporal|granularity",
      "resolution": "kept|discarded|modified",
      "resolved_triple": ["head", "relation", "tail"] or null,
      "reason": "explanation"
    }
  ],
  "unresolved_conflicts": [],
  "summary": "brief summary"
}
"""
)

# ---------------------------------------------------------------------------
# RAG QA (legacy freeform Thought:/Answer:)
# ---------------------------------------------------------------------------

RAG_QA_SYSTEM = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages '
    'and corresponding questions meticulously. Your response starts after "Thought: ", where '
    "you methodically break down the reasoning process. Conclude with \"Answer: \" to present "
    "a concise, definitive response."
)

RAG_QA_USER_TEMPLATE = Template(
    """$context

Question: $question
Thought: """
)

# ---------------------------------------------------------------------------
# RAG QA (structured JSON — default for API consumers)
# ---------------------------------------------------------------------------

RAG_QA_STRUCTURED_SYSTEM = """You are a reading-comprehension assistant for a GraphRAG system.
Analyze the numbered passages and answer the question.

Rules:
- Ground every claim in the passages. If the passages do not contain enough information, say so clearly in "answer".
- EVERY answer MUST reference document sources. Each passage header includes "Source: <filename>".
- In "answer", cite supporting passages with [n] markers (1-based) and name the Source filename(s).
- "citations" must be 1-based passage numbers that support the answer (empty list only if no passage applies).
- "sources" must list {passage, file_path} for every cited passage; use the Source filename from the header.
- "confidence" is one of: high, medium, low.
- Respond with a single JSON object only. No markdown fences, no prose outside JSON.

Required JSON shape:
{
  "thought": "<brief reasoning grounded in the passages and their Source filenames>",
  "answer": "<concise definitive answer that names Source filename(s) and uses [n] citations>",
  "citations": [<int>, ...],
  "sources": [{"passage": <int>, "file_path": "<Source filename>"}],
  "confidence": "high|medium|low"
}
"""

RAG_QA_STRUCTURED_USER_TEMPLATE = Template(
    """Passages:
$context

Question: $question

Respond with JSON only. Always cite Source filenames in the answer.
"""
)

RAG_QA_STRUCTURED_BYPASS_SYSTEM = """You are a helpful assistant.
Answer the user question and return a single JSON object only (no markdown fences).
There is no retrieved document corpus in bypass mode, so sources/citations stay empty.

Required JSON shape:
{
  "thought": "<brief reasoning>",
  "answer": "<concise definitive answer>",
  "citations": [],
  "sources": [],
  "confidence": "high|medium|low"
}
"""


def _numbered_context(
    docs: list[str], sources: list[str] | None = None
) -> str:
    """Format passages as ``[Passage N | Source: …]`` blocks for grounding."""
    parts: list[str] = []
    source_list = list(sources or [])
    for i, doc in enumerate(docs, start=1):
        text = str(doc or "").strip()
        if not text:
            continue
        src = ""
        if i - 1 < len(source_list):
            src = str(source_list[i - 1] or "").strip()
        if src:
            parts.append(f"[Passage {i} | Source: {src}]\n{text}")
        else:
            parts.append(f"[Passage {i} | Source: unknown]\n{text}")
    return "\n\n".join(parts)


def parse_structured_qa(raw: str) -> dict[str, object]:
    """Parse a structured QA LLM response into normalized fields.

    Returns keys: ``answer``, ``thought``, ``citations``, ``sources``,
    ``confidence``, ``structured`` (bool), ``raw``.
    Falls back to freeform text when JSON is missing or invalid.
    """
    from memgraphrag.utils.json_llm import extract_json_object

    text = str(raw or "").strip()
    data = extract_json_object(text)
    if not data:
        # Legacy Thought:/Answer: fallback
        answer = text
        thought: str | None = None
        if "Answer:" in text:
            before, _, after = text.partition("Answer:")
            answer = after.strip()
            if "Thought:" in before:
                thought = before.split("Thought:", 1)[-1].strip() or None
            elif before.strip():
                thought = before.strip()
        return {
            "answer": answer,
            "thought": thought,
            "citations": [],
            "sources": [],
            "confidence": None,
            "structured": False,
            "raw": text,
        }

    citations_raw = data.get("citations") or []
    citations: list[int] = []
    if isinstance(citations_raw, list):
        for item in citations_raw:
            try:
                citations.append(int(item))
            except (TypeError, ValueError):
                continue

    sources_out: list[dict[str, object]] = []
    sources_raw = data.get("sources") or []
    if isinstance(sources_raw, list):
        for item in sources_raw:
            if not isinstance(item, dict):
                continue
            passage = item.get("passage")
            file_path = item.get("file_path") or item.get("source") or item.get("name")
            try:
                passage_i = int(passage) if passage is not None else None
            except (TypeError, ValueError):
                passage_i = None
            if file_path is None and passage_i is None:
                continue
            sources_out.append(
                {
                    "passage": passage_i,
                    "file_path": str(file_path).strip() if file_path is not None else "",
                }
            )

    confidence = data.get("confidence")
    if confidence is not None:
        confidence = str(confidence).strip().lower() or None
        if confidence not in {"high", "medium", "low"}:
            confidence = None

    answer = data.get("answer")
    thought_val = data.get("thought")
    return {
        "answer": str(answer).strip() if answer is not None else text,
        "thought": str(thought_val).strip() if thought_val is not None else None,
        "citations": citations,
        "sources": sources_out,
        "confidence": confidence,
        "structured": True,
        "raw": text,
    }


def render_ner(passage: str) -> tuple[str, str]:
    """Return (system, user) prompts for NER."""
    return NER_SYSTEM, NER_USER_TEMPLATE.substitute(passage=passage)


def render_triple_extraction(passage: str, named_entities: list[str]) -> tuple[str, str]:
    """Return (system, user) prompts for triple extraction."""
    import json

    named_entity_json = json.dumps({"named_entities": named_entities}, ensure_ascii=False)
    return TRIPLE_EXTRACTION_SYSTEM, TRIPLE_EXTRACTION_USER_TEMPLATE.substitute(
        passage=passage, named_entity_json=named_entity_json
    )


def render_rag_qa(question: str, docs: list[str]) -> tuple[str, str]:
    """Return (system, user) prompts for legacy freeform RAG QA."""
    context = "\n\n".join(docs)
    return RAG_QA_SYSTEM, RAG_QA_USER_TEMPLATE.substitute(context=context, question=question)


def render_rag_qa_structured(
    question: str,
    docs: list[str],
    sources: list[str] | None = None,
) -> tuple[str, str]:
    """Return (system, user) prompts for structured JSON RAG QA."""
    context = _numbered_context(docs, sources=sources)
    return (
        RAG_QA_STRUCTURED_SYSTEM,
        RAG_QA_STRUCTURED_USER_TEMPLATE.substitute(context=context, question=question),
    )
