"""Tests for the LightRAG sidecar importer.

The importer exists so the RFE corpus does not have to be re-parsed: LightRAG already
paid ~36 h of Docling+VLM for it. What it must guarantee is that nothing the LLM
cannot read survives into the index — a raw `<table format="json">[[…]]` extracts as
noise, and a `<drawing/>` tag extracts as nothing at all.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "import_lightrag_parsed.py"


@pytest.fixture(scope="module")
def importer():
    """Load the script as a module (it has no import-time side effects).

    Registered in ``sys.modules`` first: ``@dataclass`` resolves annotations through
    ``sys.modules[cls.__module__]``, which is absent for a module loaded by path.
    """
    name = "_import_lightrag_parsed_probe"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def test_json_matrix_becomes_a_markdown_table(importer) -> None:
    payload = json.dumps(
        [
            ["Cas", "Qui vend ?", "Qui facture ?"],
            ["9", "Vendeur", "Vendeur"],
            ["17b", "Vendeur", "Marketplace"],
        ]
    )
    rendered = importer.table_to_markdown(payload)

    assert rendered is not None
    lines = rendered.splitlines()
    assert lines[0] == "| Cas | Qui vend ? | Qui facture ? |"
    assert set(lines[1]) <= {"|", "-"}, "second line must be the separator"
    assert "| 17b | Vendeur | Marketplace |" in rendered


def test_ragged_rows_are_padded_not_dropped(importer) -> None:
    """A short row must not shift the remaining cells into the wrong column."""
    rendered = importer.table_to_markdown(json.dumps([["a", "b", "c"], ["1"], ["x", "y"]]))
    assert rendered is not None
    for line in rendered.splitlines():
        assert line.count("|") == 4


def test_unusable_table_payload_returns_none(importer) -> None:
    assert importer.table_to_markdown("not json") is None
    assert importer.table_to_markdown("[]") is None


def test_drawing_becomes_prose_with_ocr(importer) -> None:
    entry = {
        "llm_analyze_result": {
            "status": "success",
            "name": "schema des 4 coins",
            "description": "Un schéma reliant vendeur, PA-E, PA-R et acheteur.",
        },
        "ocr_texts": ["PA-E", "PA-R"],
    }
    text = importer.drawing_to_text(entry)

    assert text is not None
    assert text.startswith("[Image]")
    assert "schéma reliant vendeur" in text
    assert "PA-E" in text and "PA-R" in text


def test_drawing_skipped_by_the_vlm_yields_nothing(importer) -> None:
    """Sub-64px images are bullets and rules; they carry no knowledge.

    1 706 of the benchmark's 1 997 drawings are in this state.
    """
    entry = {
        "llm_analyze_result": {
            "status": "skipped",
            "message": "image width or height is smaller than 64px",
        }
    }
    assert importer.drawing_to_text(entry) is None
    assert importer.drawing_to_text(None) is None


def test_transform_block_leaves_no_markup_behind(importer) -> None:
    stats = importer.Stats()
    content = (
        "## Vue d'ensemble\n"
        '<drawing id="im-1" format="png" path="x.png" src="" />\n'
        '<table id="tb-1" format="json">[["Cas","Qui vend ?"],["9","Vendeur"]]</table>\n'
        "Texte de conclusion."
    )
    drawings = {
        "im-1": {
            "llm_analyze_result": {
                "status": "success",
                "name": "flux",
                "description": "Le vendeur émet la facture.",
            }
        }
    }

    out = importer.transform_block(content, drawings, {}, {}, stats)

    assert "<drawing" not in out and "<table" not in out
    assert "| Cas | Qui vend ? |" in out
    assert "Le vendeur émet la facture." in out
    assert "Texte de conclusion." in out
    assert stats.tables_inlined == 1
    assert stats.drawings_inlined == 1


def test_table_column_spacing_is_preserved(importer) -> None:
    """Intra-line spacing is the only thing holding plain-text columns apart.

    The corpus profile measured that no PDF uses border characters, so collapsing
    runs of spaces would destroy 22-24 % of the lines of the two largest documents.
    """
    stats = importer.Stats()
    content = "Cadre B        Cadre S\nLivraison      Prestation"
    out = importer.transform_block(content, {}, {}, {}, stats)
    assert "Cadre B        Cadre S" in out


def test_kerned_title_is_reglued(importer) -> None:
    text, fixed = importer.unkern("D E C L A R A T I O N numérique")
    assert "DECLARATION" in text
    assert fixed == 1


def test_unkern_leaves_normal_french_alone(importer) -> None:
    """Single-letter French words must survive: 'a', 'y', and initials."""
    for phrase in (
        "il y a une facture",
        "la reforme de la facture electronique",
        "TVA a 20 %",
    ):
        out, _ = importer.unkern(phrase)
        assert out == phrase, f"mangled: {phrase!r} -> {out!r}"
