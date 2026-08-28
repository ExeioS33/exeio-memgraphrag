"""The evaluation harness must chunk corpus documents before indexing.

`ainsert` takes chunks already cut — only the file-ingestion pipeline chunks for
you — and two of the four datasets ship their corpus as a single plain-text file.
`medical.txt` is ~221k tokens, so passing documents straight through sent one
passage far past any context window and the run aborted on a provider 400.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"


def _load_chunker():
    """Import only `_chunk_corpus`, without executing the script's module body.

    The script calls `load_env_file` at import, which would inject the developer's
    .env — real provider keys included — into the test process.
    """
    spec = importlib.util.spec_from_file_location("_eval_probe", SCRIPT)
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_chunk_corpus"
    )
    module = importlib.util.module_from_spec(spec)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SCRIPT), "exec"), module.__dict__)
    return module._chunk_corpus


class _Doc:
    """Mirrors evaluation.CorpusDocument: a `text` attribute plus `to_chunk()`."""

    def __init__(self, title: str, text: str) -> None:
        self.title = title
        self.text = text

    def to_chunk(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


def test_long_document_is_split_into_bounded_chunks() -> None:
    chunk_corpus = _load_chunker()
    # ~40k words: far beyond any single-request budget, as medical.txt is.
    doc = _Doc("bigdoc", " ".join(f"word{i}" for i in range(40000)))

    chunks = chunk_corpus([doc], 1200, 100)

    assert len(chunks) > 1, "a huge document must not stay a single chunk"
    from memgraphrag.utils.tokenizer import TiktokenTokenizer

    tokenizer = TiktokenTokenizer()
    sizes = [len(tokenizer.encode(c)) for c in chunks]
    # Allow the title prefix on top of the window, but nothing near a context limit.
    assert max(sizes) < 1200 * 2, f"chunk overflowed the window: {max(sizes)}"


def test_every_chunk_keeps_its_document_title() -> None:
    """Retrieval scoring recovers the source document from the passage's first line."""
    chunk_corpus = _load_chunker()
    doc = _Doc("Some Title", " ".join(f"w{i}" for i in range(5000)))

    chunks = chunk_corpus([doc], 400, 40)

    assert len(chunks) > 1
    assert all(c.startswith("Some Title") for c in chunks)


def test_empty_documents_are_dropped() -> None:
    chunk_corpus = _load_chunker()
    assert chunk_corpus([_Doc("t", "   ")], 400, 40) == []
