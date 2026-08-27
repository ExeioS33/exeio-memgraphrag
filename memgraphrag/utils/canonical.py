"""Canonical surface form for entity and relation labels.

Why this exists
---------------
The engine used to fold surfaces with `strip().lower()` only. On an English corpus
that is nearly enough; on a French one it is not. `Réforme de la Facture
Électronique`, `REFORME DE LA FACTURE ELECTRONIQUE` and `réforme de la facture
electronique` all name the same thing and all became **different entities**, each
with its own vector, its own graph node and its own schema — which is exactly what
splits the schema layer and, past ONTOLOGY_MAX_DEACTIVATION_RATIO, switches the
ontology filter off.

The accent-folding logic already existed in ``memgraphrag/evaluation/normalization``
but was imported only by the offline metrics, never by the engine.

Only the *matching key* is folded. Display text keeps its accents: an answer must
read "Réforme de la Facture Électronique", not "reforme de la facture electronique".
"""

from __future__ import annotations

import re
import unicodedata

#: Typographic characters that models and PDF extractors emit interchangeably.
_FOLD = {
    **dict.fromkeys(map(ord, "‐‑‒–—―−"), "-"),
    **dict.fromkeys(map(ord, "‘’‛′`"), "'"),
    **dict.fromkeys(map(ord, "“”‟″"), '"'),
    ord(" "): " ",
    ord(" "): " ",
    ord("…"): "...",
    ord("œ"): "oe",
    ord("Œ"): "OE",
    ord("æ"): "ae",
    ord("Æ"): "AE",
}

_WHITESPACE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """Drop combining marks: ``é`` -> ``e``, ``ç`` -> ``c``."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def canonical_key(text: str) -> str:
    """Fold a label to its matching key.

    NFKC, typographic folding, accent stripping, case folding and whitespace
    collapsing — in that order, so that ligatures resolve before accents are removed.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", str(text)).translate(_FOLD)
    folded = strip_accents(folded).casefold()
    return _WHITESPACE.sub(" ", folded).strip()


def canonical_triple(triple) -> tuple[str, str, str]:
    """Canonical matching key for a (head, relation, tail) triple."""
    head, relation, tail = (triple[0], triple[1], triple[2])
    return (canonical_key(head), canonical_key(relation), canonical_key(tail))
