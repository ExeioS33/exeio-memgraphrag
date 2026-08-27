"""Loaders for the four benchmark datasets used by the MemGraphRAG paper.

The research checkout ships them under ``MemGraphRAG/dataset/`` in two different
shapes, both handled here:

* **flat** — ``hotpotqa/hotpotqa.json`` and
  ``2wikimutlhopqa/2wikimultihopqa.json`` are lists of 1000 records with
  ``_id`` / ``question`` / ``answer`` / ``supporting_facts`` / ``context``.
* **grouped** — ``musique/musique.json`` and ``medical/question.json`` are a
  one-element list wrapping ``{"source": ..., "questions": {"type1": [...]}}``;
  MuSiQue records carry ``paragraphs`` with an ``is_supporting`` flag and
  ``answer_aliases``, medical records carry ``source`` / ``evidence``.

Nothing in this repository depends on that checkout being present: every entry
point raises :class:`DatasetUnavailableError` with the path it looked at, so a
developer without the 58 MB research tree gets one clear sentence instead of a
``FileNotFoundError`` from inside a metric.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Overrides the dataset location; the default assumes the research checkout is
#: a sibling of this repository, which is how the workspace is laid out.
DATASET_ROOT_ENV = "MEMGRAPHRAG_DATASET_ROOT"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = _REPO_ROOT.parent / "MemGraphRAG" / "dataset"


class DatasetUnavailableError(RuntimeError):
    """The requested dataset file is not on disk (research checkout missing)."""


class DatasetFormatError(ValueError):
    """A dataset file was found but does not have the structure documented above."""


@dataclass(frozen=True)
class EvaluationExample:
    """One scorable question.

    ``gold_docs`` holds *document identities* (Wikipedia titles, or the source
    label for the medical set), not passage text: the retrieval metrics compare
    identities, and a passage-text comparison would depend on the chunker.
    """

    id: str
    question: str
    gold_answers: list[str]
    gold_docs: list[str] = field(default_factory=list)
    dataset: str = ""
    question_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "gold_answers": list(self.gold_answers),
            "gold_docs": list(self.gold_docs),
            "dataset": self.dataset,
            "question_type": self.question_type,
        }


@dataclass(frozen=True)
class CorpusDocument:
    """One indexable document of a benchmark corpus."""

    title: str
    text: str

    def to_chunk(self) -> str:
        """Title-prefixed text, so a retrieved passage can be traced to its title."""
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset lives and which of the two record shapes it uses."""

    name: str
    questions: str
    corpus_json: str | None = None
    corpus_text: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "hotpotqa": DatasetSpec(
        name="hotpotqa",
        questions="hotpotqa/hotpotqa.json",
        corpus_json="hotpotqa/hotpotqa_corpus.json",
        corpus_text="hotpotqa/hotpotqa.txt",
    ),
    # Directory name is misspelled in the research checkout ("mutlhop"); kept
    # verbatim because renaming it here would just fail to find the files.
    "2wikimultihopqa": DatasetSpec(
        name="2wikimultihopqa",
        questions="2wikimutlhopqa/2wikimultihopqa.json",
        corpus_json="2wikimutlhopqa/2wikimultihopqa_corpus.json",
        corpus_text="2wikimutlhopqa/2wikimultihopqa.txt",
    ),
    "musique": DatasetSpec(
        name="musique",
        questions="musique/musique.json",
        corpus_json="musique/musique_corpus.json",
        corpus_text="musique/musique.txt",
    ),
    # The medical set ships no corpus JSON, only one flat text file.
    "medical": DatasetSpec(
        name="medical",
        questions="medical/question.json",
        corpus_json=None,
        corpus_text="medical/medical.txt",
    ),
}

#: Aliases so a caller may use the directory spelling of the research checkout.
DATASET_ALIASES = {"2wikimutlhopqa": "2wikimultihopqa", "2wiki": "2wikimultihopqa"}


def resolve_dataset(name: str) -> DatasetSpec:
    """Look up a dataset spec by name or alias."""
    key = DATASET_ALIASES.get(name.strip().lower(), name.strip().lower())
    try:
        return DATASETS[key]
    except KeyError:
        raise DatasetUnavailableError(
            f"unknown dataset {name!r}; known datasets: {', '.join(sorted(DATASETS))}"
        ) from None


def dataset_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the dataset root: explicit argument, then env var, then default."""
    if explicit:
        return Path(explicit).expanduser()
    from_env = os.getenv(DATASET_ROOT_ENV)
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_DATASET_ROOT


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise DatasetUnavailableError(
            f"{what} not found at {path}. The benchmark datasets live in the research "
            f"checkout (MemGraphRAG/dataset/); point {DATASET_ROOT_ENV} at it or pass "
            "--dataset-root."
        )
    return path


def available_datasets(root: str | os.PathLike[str] | None = None) -> list[str]:
    """Names of the datasets whose question file is actually present under ``root``."""
    base = dataset_root(root)
    return [spec.name for spec in DATASETS.values() if (base / spec.questions).exists()]


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _iter_records(payload: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield ``(question_type, record)`` from either dataset shape.

    Detection is by content, not by dataset name: the two shapes are told apart
    by whether a record carries a ``questions`` mapping, so a dataset that
    changes shape upstream still loads instead of loading *wrongly*.
    """
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise DatasetFormatError(f"expected a list or object at the top level, got {type(payload)}")
    for entry in payload:
        if not isinstance(entry, dict):
            raise DatasetFormatError(f"expected objects in the dataset list, got {type(entry)}")
        grouped = entry.get("questions")
        if isinstance(grouped, dict):
            for question_type, records in grouped.items():
                for record in records or []:
                    yield str(question_type), record
        elif isinstance(grouped, list):
            for record in grouped:
                yield "", record
        else:
            yield str(entry.get("type") or ""), entry


def _gold_docs(record: dict[str, Any]) -> list[str]:
    """Supporting-document titles, whichever way this record spells them."""
    titles: list[str] = []
    supporting = record.get("supporting_facts")
    if isinstance(supporting, list):
        for fact in supporting:
            # HotpotQA / 2Wiki: [title, sentence_index]. Some dumps use dicts.
            if isinstance(fact, (list, tuple)) and fact:
                titles.append(str(fact[0]))
            elif isinstance(fact, dict) and fact.get("title"):
                titles.append(str(fact["title"]))
    paragraphs = record.get("paragraphs")
    if isinstance(paragraphs, list):
        titles.extend(
            str(para.get("title", ""))
            for para in paragraphs
            if isinstance(para, dict) and para.get("is_supporting")
        )
    if not titles and record.get("source"):
        # Medical records name their source document rather than listing facts.
        titles.append(str(record["source"]))
    seen: set[str] = set()
    unique: list[str] = []
    for title in titles:
        if title and title not in seen:
            seen.add(title)
            unique.append(title)
    return unique


def _gold_answers(record: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    answer = record.get("answer")
    if isinstance(answer, list):
        answers.extend(str(item) for item in answer)
    elif answer is not None:
        answers.append(str(answer))
    aliases = record.get("answer_aliases")
    if isinstance(aliases, list):
        answers.extend(str(alias) for alias in aliases)
    return [item for item in dict.fromkeys(answers) if item.strip()]


def _to_example(
    record: dict[str, Any],
    dataset: str,
    question_type: str,
    index: int,
) -> EvaluationExample:
    question = str(record.get("question") or "").strip()
    if not question:
        raise DatasetFormatError(
            f"{dataset} record #{index} has no 'question' field (keys: {sorted(record)})"
        )
    return EvaluationExample(
        id=str(record.get("_id") or record.get("id") or f"{dataset}-{index}"),
        question=question,
        gold_answers=_gold_answers(record),
        gold_docs=_gold_docs(record),
        dataset=dataset,
        question_type=str(record.get("type") or question_type or ""),
        metadata={
            key: record[key]
            for key in ("level", "evidence", "source", "answerable")
            if key in record
        },
    )


def load_questions(
    name: str,
    root: str | os.PathLike[str] | None = None,
    limit: int | None = None,
    question_types: Sequence[str] | None = None,
) -> list[EvaluationExample]:
    """Load a dataset's questions in file order.

    ``limit`` truncates rather than samples, so two runs of the same command see
    the same questions; use :func:`sample_examples` when a random subset is
    wanted, and record its seed.
    """
    spec = resolve_dataset(name)
    path = _require(dataset_root(root) / spec.questions, f"{spec.name} questions")
    wanted = {str(item) for item in question_types} if question_types else None
    examples: list[EvaluationExample] = []
    for index, (question_type, record) in enumerate(_iter_records(_load_json(path))):
        if wanted is not None and question_type not in wanted:
            continue
        examples.append(_to_example(record, spec.name, question_type, index))
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise DatasetFormatError(f"no questions loaded from {path} (filters: {question_types})")
    return examples


def load_corpus(
    name: str,
    root: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> list[CorpusDocument]:
    """Load a dataset's corpus documents.

    The medical set ships no corpus JSON, only ``medical.txt``; it is returned as
    a single document so the caller's chunker decides how to split it, rather
    than this loader inventing a segmentation the dataset never specified.
    """
    spec = resolve_dataset(name)
    base = dataset_root(root)
    if spec.corpus_json:
        path = _require(base / spec.corpus_json, f"{spec.name} corpus")
        payload = _load_json(path)
        if not isinstance(payload, list):
            raise DatasetFormatError(f"expected a list of documents in {path}")
        docs = [
            CorpusDocument(title=str(item.get("title") or ""), text=str(item.get("text") or ""))
            for item in payload
            if isinstance(item, dict)
        ]
    elif spec.corpus_text:
        path = _require(base / spec.corpus_text, f"{spec.name} corpus")
        docs = [CorpusDocument(title=spec.name, text=path.read_text(encoding="utf-8"))]
    else:  # pragma: no cover - every spec declares one of the two
        raise DatasetUnavailableError(f"{spec.name} declares no corpus file")
    docs = [doc for doc in docs if doc.text.strip()]
    return docs[:limit] if limit is not None else docs


def sample_examples(
    examples: Sequence[EvaluationExample],
    size: int,
    seed: int = 0,
) -> list[EvaluationExample]:
    """Reproducible random subset, kept in the dataset's own order.

    Sampling is seeded and the order preserved because a variance campaign has to
    vary the engine, not the question set: a different subset per run would make
    the standard deviation measure the sampler.
    """
    if size >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(examples)), size))
    return [examples[index] for index in indices]
