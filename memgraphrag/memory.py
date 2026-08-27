"""
Build a three-layer Memory structure with inter-layer connections:
1. Schema layer: Stores ontology (type triples), frequency, vectors, lower-level fact indices
2. Fact layer: Stores triples, vectors, upper-level schema indices, lower-level passage indices
3. Passage layer: Stores chunks, vectors, upper-level fact indices

Inter-layer relationships:
    Schema --1:N--> Fact --N:M--> Passage

Raw OpenIE outputs (``extracted_triples`` only) can be loaded with
``ThreeLayerMemory.build_from_raw_openie_results`` so ``schema_layer`` stays empty.

Provenance: adapted from MemGraphRAG/code/src/Memory.py.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional CLI defaults (no hardcoded absolute paths).
DEFAULT_FILTERED_PATH: Optional[str] = None
DEFAULT_MEMORY_OUTPUT_PATH: Optional[str] = None


@dataclass
class SchemaNode:
    """Schema layer node: stores ontology information."""

    idx: int  # Unique index
    content: Tuple[str, str, str]  # ontology: (head_type, relation, tail_type)
    frequency: int = 0  # Frequency of this ontology
    embedding: Optional[List[float]] = None  # Vector representation
    fact_indices: List[int] = field(default_factory=list)  # Lower-level fact index list


@dataclass
class FactNode:
    """Fact layer node: stores triple information."""

    idx: int  # Unique index
    content: Tuple[str, str, str]  # Triple: (head, relation, tail)
    frequency: int = 0  # Distinct chunks (passages) this fact appears in
    embedding: Optional[List[float]] = None  # Vector representation
    schema_idx: int = -1  # Upper-level schema index (unique)
    passage_indices: List[int] = field(default_factory=list)  # Lower-level passage index list


@dataclass
class PassageNode:
    """Passage layer node: stores chunk information."""

    idx: int  # Unique index
    chunk_id: str  # Original chunk id
    content: str  # Passage text
    embedding: Optional[List[float]] = None  # Vector representation
    fact_indices: List[int] = field(default_factory=list)  # Upper-level fact index list
    modality: str = "text"  # Extension point for VLM (e.g. "image", "audio")


class ThreeLayerMemory:
    """Three-layer Memory structure."""

    def __init__(self) -> None:
        # Three layer lists
        self.schema_layer: List[SchemaNode] = []
        self.fact_layer: List[FactNode] = []
        self.passage_layer: List[PassageNode] = []

        # Mappings for deduplication and fast lookup
        self._schema_to_idx: Dict[Tuple[str, str, str], int] = {}
        self._fact_to_idx: Dict[Tuple[str, str, str], int] = {}
        self._chunk_id_to_idx: Dict[str, int] = {}

    def _get_or_create_schema(self, ontology: Tuple[str, str, str]) -> int:
        """Get or create schema node, return index."""
        if ontology in self._schema_to_idx:
            return self._schema_to_idx[ontology]

        idx = len(self.schema_layer)
        node = SchemaNode(idx=idx, content=ontology)
        self.schema_layer.append(node)
        self._schema_to_idx[ontology] = idx
        return idx

    def _get_or_create_fact(self, triple: Tuple[str, str, str], schema_idx: int) -> int:
        """Get or create fact node, return index."""
        if triple in self._fact_to_idx:
            return self._fact_to_idx[triple]

        idx = len(self.fact_layer)
        node = FactNode(idx=idx, content=triple, schema_idx=schema_idx)
        self.fact_layer.append(node)
        self._fact_to_idx[triple] = idx
        return idx

    def _get_or_create_passage(self, chunk_id: str, passage_text: str) -> int:
        """Get or create passage node, return index."""
        if chunk_id in self._chunk_id_to_idx:
            return self._chunk_id_to_idx[chunk_id]

        idx = len(self.passage_layer)
        node = PassageNode(idx=idx, chunk_id=chunk_id, content=passage_text)
        self.passage_layer.append(node)
        self._chunk_id_to_idx[chunk_id] = idx
        return idx

    @staticmethod
    def _triple_tuple_from_openie_item(item: Any) -> Optional[Tuple[str, str, str]]:
        """
        Normalize one element from extracted_triples to (h, r, t).

        Matches common OpenIE payloads: processed_triple, raw_triple, triple_str,
        or a bare length-3 list/tuple.
        """
        if item is None:
            return None
        if isinstance(item, (list, tuple)) and len(item) == 3:
            try:
                return (str(item[0]), str(item[1]), str(item[2]))
            except Exception:
                return None
        if isinstance(item, dict):
            for key in ("processed_triple", "triple", "raw_triple"):
                triple = item.get(key)
                if isinstance(triple, (list, tuple)) and len(triple) == 3:
                    try:
                        return (str(triple[0]), str(triple[1]), str(triple[2]))
                    except Exception:
                        pass
            ts = item.get("triple_str")
            if isinstance(ts, str):
                try:
                    t = ast.literal_eval(ts.strip())
                    if isinstance(t, (list, tuple)) and len(t) == 3:
                        return (str(t[0]), str(t[1]), str(t[2]))
                except Exception:
                    pass
        return None

    def build_from_raw_openie_results(self, data: Dict[str, Any]) -> None:
        """
        Build fact + passage layers from raw OpenIE JSON (no schema / ontology).

        Expects the same ``docs`` list shape as ner OpenIE outputs: each doc has
        ``idx``, ``passage``, and ``extracted_triples`` (list of dicts or 3-tuples).

        Does not populate ``schema_layer``. Every ``FactNode`` has ``schema_idx == -1``.

        Prefer a freshly constructed ``ThreeLayerMemory`` (``schema_layer`` is not cleared).
        """
        docs = data.get("docs", [])
        logger.info("Building two-layer memory (no schema) from %d docs...", len(docs))

        for doc in docs:
            chunk_id = doc.get("idx", "")
            passage_text = doc.get("passage", "")
            triple_entries = doc.get("extracted_triples") or []

            valid_triples: List[Tuple[str, str, str]] = []
            if isinstance(triple_entries, list):
                for item in triple_entries:
                    triple_tuple = self._triple_tuple_from_openie_item(item)
                    if triple_tuple is not None:
                        valid_triples.append(triple_tuple)

            # Create the passage even with no usable triple. Skipping it used to make
            # the chunk absent from the passage layer, hence from chunks_vdb, hence
            # unreachable even by the dense fallback: a table of figures or a cover
            # page silently disappeared from the corpus. An orphan passage still
            # answers dense queries; a missing one answers nothing.
            passage_idx = self._get_or_create_passage(chunk_id, passage_text)
            if not valid_triples:
                continue

            for triple_tuple in valid_triples:
                fact_idx = self._get_or_create_fact(triple_tuple, schema_idx=-1)

                if passage_idx not in self.fact_layer[fact_idx].passage_indices:
                    self.fact_layer[fact_idx].passage_indices.append(passage_idx)
                if fact_idx not in self.passage_layer[passage_idx].fact_indices:
                    self.passage_layer[passage_idx].fact_indices.append(fact_idx)

        for fact_node in self.fact_layer:
            fact_node.frequency = len(fact_node.passage_indices)

        logger.info("Built memory structure:")
        logger.info(
            "  Schema layer: %d unique ontologies (expected 0)",
            len(self.schema_layer),
        )
        logger.info("  Fact layer:   %d unique triples", len(self.fact_layer))
        logger.info("  Passage layer: %d unique chunks", len(self.passage_layer))

    def build_from_openie_results(self, data: Dict[str, Any]) -> None:
        """
        Build three-layer structure from openie_results_with_ontology_filtered.json.
        """
        docs = data.get("docs", [])
        logger.info("Building memory from %d docs...", len(docs))

        for doc in docs:
            chunk_id = doc.get("idx", "")
            passage_text = doc.get("passage", "")
            triple_ont_map = doc.get("extracted_triple_ontology") or {}

            if not isinstance(triple_ont_map, dict) or not triple_ont_map:
                continue

            # 1. First create passage node
            passage_idx = self._get_or_create_passage(chunk_id, passage_text)

            # 2. Traverse all (triple_key, ontology) in this doc
            for triple_key, ontology in triple_ont_map.items():
                # Parse triple_key -> tuple
                try:
                    triple_tuple = ast.literal_eval(triple_key)
                    if not (isinstance(triple_tuple, tuple) and len(triple_tuple) == 3):
                        continue
                except Exception:
                    continue

                # Parse ontology -> tuple
                if not (isinstance(ontology, list) and len(ontology) == 3):
                    continue
                ontology_tuple = tuple(ontology)

                # 3. Create or get schema node
                schema_idx = self._get_or_create_schema(ontology_tuple)

                # 4. Create or get fact node
                fact_idx = self._get_or_create_fact(triple_tuple, schema_idx)

                # 5. Establish inter-layer index relationships
                # Schema -> Fact (lower-level index)
                if fact_idx not in self.schema_layer[schema_idx].fact_indices:
                    self.schema_layer[schema_idx].fact_indices.append(fact_idx)

                # Fact -> Passage (lower-level index)
                if passage_idx not in self.fact_layer[fact_idx].passage_indices:
                    self.fact_layer[fact_idx].passage_indices.append(passage_idx)

                # Passage -> Fact (upper-level index)
                if fact_idx not in self.passage_layer[passage_idx].fact_indices:
                    self.passage_layer[passage_idx].fact_indices.append(fact_idx)

        # 6. Count schema frequency (number of facts for each schema)
        for schema_node in self.schema_layer:
            schema_node.frequency = len(schema_node.fact_indices)

        # 7. Fact frequency: distinct chunks this triple was extracted from
        for fact_node in self.fact_layer:
            fact_node.frequency = len(fact_node.passage_indices)

        logger.info("Built memory structure:")
        logger.info("  Schema layer: %d unique ontologies", len(self.schema_layer))
        logger.info("  Fact layer:   %d unique triples", len(self.fact_layer))
        logger.info("  Passage layer: %d unique chunks", len(self.passage_layer))

    def get_schema_by_idx(self, idx: int) -> Optional[SchemaNode]:
        """Get schema node by index."""
        if 0 <= idx < len(self.schema_layer):
            return self.schema_layer[idx]
        return None

    def get_fact_by_idx(self, idx: int) -> Optional[FactNode]:
        """Get fact node by index."""
        if 0 <= idx < len(self.fact_layer):
            return self.fact_layer[idx]
        return None

    def get_passage_by_idx(self, idx: int) -> Optional[PassageNode]:
        """Get passage node by index."""
        if 0 <= idx < len(self.passage_layer):
            return self.passage_layer[idx]
        return None

    def get_facts_by_schema(self, schema_idx: int) -> List[FactNode]:
        """Get all facts under a schema."""
        schema = self.get_schema_by_idx(schema_idx)
        if schema is None:
            return []
        return [self.fact_layer[i] for i in schema.fact_indices]

    def get_passages_by_fact(self, fact_idx: int) -> List[PassageNode]:
        """Get all passages from which a fact was extracted."""
        fact = self.get_fact_by_idx(fact_idx)
        if fact is None:
            return []
        return [self.passage_layer[i] for i in fact.passage_indices]

    def get_facts_by_passage(self, passage_idx: int) -> List[FactNode]:
        """Get all facts extracted from a passage."""
        passage = self.get_passage_by_idx(passage_idx)
        if passage is None:
            return []
        return [self.fact_layer[i] for i in passage.fact_indices]

    def get_schema_by_fact(self, fact_idx: int) -> Optional[SchemaNode]:
        """Get the schema corresponding to a fact."""
        fact = self.get_fact_by_idx(fact_idx)
        if fact is None:
            return None
        return self.get_schema_by_idx(fact.schema_idx)

    def link_fact_to_schema(self, fact_idx: int, ontology: Tuple[str, str, str]) -> int:
        """Link a fact to a schema ontology triple; return schema index."""
        if not (0 <= fact_idx < len(self.fact_layer)):
            raise IndexError(f"fact_idx out of range: {fact_idx}")
        ontology_tuple = (str(ontology[0]), str(ontology[1]), str(ontology[2]))
        schema_idx = self._get_or_create_schema(ontology_tuple)
        fact = self.fact_layer[fact_idx]
        old_idx = fact.schema_idx
        if old_idx >= 0 and old_idx != schema_idx and old_idx < len(self.schema_layer):
            old = self.schema_layer[old_idx]
            if fact_idx in old.fact_indices:
                old.fact_indices.remove(fact_idx)
            old.frequency = len(old.fact_indices)
        fact.schema_idx = schema_idx
        if fact_idx not in self.schema_layer[schema_idx].fact_indices:
            self.schema_layer[schema_idx].fact_indices.append(fact_idx)
        self.schema_layer[schema_idx].frequency = len(self.schema_layer[schema_idx].fact_indices)
        return schema_idx

    def recompute_schema_frequencies(self) -> None:
        """Refresh schema.frequency from fact_indices length."""
        for schema in self.schema_layer:
            schema.frequency = len(schema.fact_indices)

    def filter_schemas_by_frequency(self, min_frequency: int) -> dict[str, int]:
        """Drop schemas below ``min_frequency`` and reindex survivors.

        Facts linked to dropped schemas get ``schema_idx = -1``.
        ``min_frequency <= 0`` disables filtering.
        """
        if min_frequency <= 0 or not self.schema_layer:
            return {
                "kept": len(self.schema_layer),
                "dropped": 0,
                "min_frequency": min_frequency,
            }

        kept_nodes: List[SchemaNode] = []
        old_to_new: Dict[int, int] = {}
        dropped = 0
        for schema in self.schema_layer:
            schema.frequency = len(schema.fact_indices)
            if schema.frequency >= min_frequency:
                new_idx = len(kept_nodes)
                old_to_new[schema.idx] = new_idx
                schema.idx = new_idx
                kept_nodes.append(schema)
            else:
                dropped += 1

        for fact in self.fact_layer:
            if fact.schema_idx < 0:
                continue
            if fact.schema_idx in old_to_new:
                fact.schema_idx = old_to_new[fact.schema_idx]
            else:
                fact.schema_idx = -1

        # Rebuild fact_indices on kept schemas from facts (authoritative).
        for schema in kept_nodes:
            schema.fact_indices = []
        for fact in self.fact_layer:
            if fact.schema_idx >= 0:
                kept_nodes[fact.schema_idx].fact_indices.append(fact.idx)
        for schema in kept_nodes:
            schema.frequency = len(schema.fact_indices)

        self.schema_layer = kept_nodes
        self._schema_to_idx = {s.content: s.idx for s in self.schema_layer}
        return {
            "kept": len(self.schema_layer),
            "dropped": dropped,
            "min_frequency": min_frequency,
        }

    def _rebuild_fact_indices(self) -> None:
        """Reassign sequential fact.idx and rewrite all cross-layer references."""
        old_to_new: Dict[int, int] = {}
        for new_idx, fact in enumerate(self.fact_layer):
            old_to_new[fact.idx] = new_idx
            fact.idx = new_idx

        self._fact_to_idx = {f.content: f.idx for f in self.fact_layer}

        for schema in self.schema_layer:
            schema.fact_indices = [old_to_new[i] for i in schema.fact_indices if i in old_to_new]
            schema.frequency = len(schema.fact_indices)

        for passage in self.passage_layer:
            passage.fact_indices = [old_to_new[i] for i in passage.fact_indices if i in old_to_new]

        for fact in self.fact_layer:
            fact.frequency = len(fact.passage_indices)

    def remove_fact(self, fact_idx: int) -> bool:
        """Remove one fact and rebuild positional cross-layer indices."""
        if not (0 <= fact_idx < len(self.fact_layer)):
            return False
        fact = self.fact_layer[fact_idx]
        # Detach from schema / passages before deletion.
        if 0 <= fact.schema_idx < len(self.schema_layer):
            schema = self.schema_layer[fact.schema_idx]
            if fact.idx in schema.fact_indices:
                schema.fact_indices.remove(fact.idx)
            schema.frequency = len(schema.fact_indices)
        for pidx in list(fact.passage_indices):
            if 0 <= pidx < len(self.passage_layer):
                passage = self.passage_layer[pidx]
                if fact.idx in passage.fact_indices:
                    passage.fact_indices.remove(fact.idx)
        del self.fact_layer[fact_idx]
        self._rebuild_fact_indices()
        return True

    def replace_fact(self, fact_idx: int, new_triple: Tuple[str, str, str]) -> int:
        """Replace fact content or merge into an existing identical triple.

        Returns the surviving fact index. Passage links are preserved/merged.
        """
        if not (0 <= fact_idx < len(self.fact_layer)):
            raise IndexError(f"fact_idx out of range: {fact_idx}")
        new_triple = (str(new_triple[0]), str(new_triple[1]), str(new_triple[2]))
        fact = self.fact_layer[fact_idx]
        if fact.content == new_triple:
            return fact_idx

        existing = self._fact_to_idx.get(new_triple)
        if existing is not None and existing != fact_idx:
            # Merge passages into the existing fact, then remove this one.
            target = self.fact_layer[existing]
            old_schema = fact.schema_idx
            for pidx in fact.passage_indices:
                if pidx not in target.passage_indices:
                    target.passage_indices.append(pidx)
                if 0 <= pidx < len(self.passage_layer):
                    passage = self.passage_layer[pidx]
                    if existing not in passage.fact_indices:
                        passage.fact_indices.append(existing)
            self.remove_fact(fact_idx)
            surviving = self._fact_to_idx.get(new_triple)
            if surviving is None:
                return -1
            if old_schema >= 0 and self.fact_layer[surviving].schema_idx < 0:
                schema = self.get_schema_by_idx(old_schema)
                # After remove_fact, schema indices are unchanged (only facts rebuild).
                if schema is not None:
                    self.link_fact_to_schema(surviving, schema.content)
            return surviving

        # Rename in place.
        del self._fact_to_idx[fact.content]
        fact.content = new_triple
        self._fact_to_idx[new_triple] = fact_idx
        return fact_idx

    def add_fact_with_passages(
        self,
        triple: Tuple[str, str, str],
        passage_indices: List[int],
        schema_idx: int = -1,
    ) -> int:
        """Create or reuse a fact and attach the given passages."""
        triple = (str(triple[0]), str(triple[1]), str(triple[2]))
        fact_idx = self._get_or_create_fact(triple, schema_idx)
        fact = self.fact_layer[fact_idx]
        if schema_idx >= 0 and fact.schema_idx < 0:
            fact.schema_idx = schema_idx
            if fact_idx not in self.schema_layer[schema_idx].fact_indices:
                self.schema_layer[schema_idx].fact_indices.append(fact_idx)
            self.schema_layer[schema_idx].frequency = len(
                self.schema_layer[schema_idx].fact_indices
            )
        for pidx in passage_indices:
            if pidx not in fact.passage_indices:
                fact.passage_indices.append(pidx)
            if 0 <= pidx < len(self.passage_layer):
                passage = self.passage_layer[pidx]
                if fact_idx not in passage.fact_indices:
                    passage.fact_indices.append(fact_idx)
        fact.frequency = len(fact.passage_indices)
        return fact_idx

    def to_dict(self) -> Dict[str, Any]:
        """Export to serializable dictionary."""
        return {
            "schema_layer": [
                {
                    "idx": node.idx,
                    "content": list(node.content),
                    "frequency": node.frequency,
                    "embedding": node.embedding,
                    "fact_indices": node.fact_indices,
                }
                for node in self.schema_layer
            ],
            "fact_layer": [
                {
                    "idx": node.idx,
                    "content": list(node.content),
                    "frequency": node.frequency,
                    "embedding": node.embedding,
                    "schema_idx": node.schema_idx,
                    "passage_indices": node.passage_indices,
                }
                for node in self.fact_layer
            ],
            "passage_layer": [
                {
                    "idx": node.idx,
                    "chunk_id": node.chunk_id,
                    "content": node.content,
                    "embedding": node.embedding,
                    "fact_indices": node.fact_indices,
                    "modality": node.modality,
                }
                for node in self.passage_layer
            ],
            "stats": {
                "num_schemas": len(self.schema_layer),
                "num_facts": len(self.fact_layer),
                "num_passages": len(self.passage_layer),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreeLayerMemory":
        """Restore Memory structure from dictionary."""
        memory = cls()

        # Restore schema layer
        for item in data.get("schema_layer", []):
            node = SchemaNode(
                idx=item["idx"],
                content=tuple(item["content"]),
                frequency=item.get("frequency", 0),
                embedding=item.get("embedding"),
                fact_indices=item.get("fact_indices", []),
            )
            memory.schema_layer.append(node)
            memory._schema_to_idx[node.content] = node.idx

        # Restore fact layer
        for item in data.get("fact_layer", []):
            passage_indices = item.get("passage_indices", [])
            node = FactNode(
                idx=item["idx"],
                content=tuple(item["content"]),
                frequency=item["frequency"] if "frequency" in item else len(passage_indices),
                embedding=item.get("embedding"),
                schema_idx=item.get("schema_idx", -1),
                passage_indices=passage_indices,
            )
            memory.fact_layer.append(node)
            memory._fact_to_idx[node.content] = node.idx

        # Restore passage layer
        for item in data.get("passage_layer", []):
            node = PassageNode(
                idx=item["idx"],
                chunk_id=item["chunk_id"],
                content=item["content"],
                embedding=item.get("embedding"),
                fact_indices=item.get("fact_indices", []),
                modality=item.get("modality", "text"),
            )
            memory.passage_layer.append(node)
            memory._chunk_id_to_idx[node.chunk_id] = node.idx

        return memory

    def save(self, path: str) -> None:
        """Save to JSON file."""
        logger.info("Saving memory structure to: %s", path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Saved.")

    @classmethod
    def load(cls, path: str) -> "ThreeLayerMemory":
        """Load from JSON file."""
        logger.info("Loading memory structure from: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        memory = cls.from_dict(data)
        logger.info(
            "Loaded: %d schemas, %d facts, %d passages",
            len(memory.schema_layer),
            len(memory.fact_layer),
            len(memory.passage_layer),
        )
        return memory

    def print_summary(self) -> None:
        """Log statistical summary."""
        logger.info("===== Three-Layer Memory Summary =====")
        logger.info("Schema layer (ontologies): %d", len(self.schema_layer))
        logger.info("Fact layer (triples):      %d", len(self.fact_layer))
        logger.info("Passage layer (chunks):    %d", len(self.passage_layer))

        if self.schema_layer:
            freqs = [s.frequency for s in self.schema_layer]
            logger.info(
                "Schema frequency stats: Min=%d, Max=%d, Avg=%.2f",
                min(freqs),
                max(freqs),
                sum(freqs) / len(freqs),
            )

        if self.fact_layer:
            passage_counts = [f.frequency for f in self.fact_layer]
            logger.info(
                "Fact -> Passage count stats: Min=%d, Max=%d, Avg=%.2f",
                min(passage_counts),
                max(passage_counts),
                sum(passage_counts) / len(passage_counts),
            )

        if self.passage_layer:
            fact_counts = [len(p.fact_indices) for p in self.passage_layer]
            logger.info(
                "Passage -> Fact count stats: Min=%d, Max=%d, Avg=%.2f",
                min(fact_counts),
                max(fact_counts),
                sum(fact_counts) / len(fact_counts),
            )

    def print_sample(self, n: int = 3) -> None:
        """Log sample data."""
        logger.info("===== Sample Data (top %d) =====", n)

        logger.info("--- Schema Layer ---")
        for schema in self.schema_layer[:n]:
            logger.debug(
                "  [%d] %s, freq=%d, facts=%s%s",
                schema.idx,
                schema.content,
                schema.frequency,
                schema.fact_indices[:5],
                "..." if len(schema.fact_indices) > 5 else "",
            )

        logger.info("--- Fact Layer ---")
        for fact in self.fact_layer[:n]:
            logger.debug(
                "  [%d] %s, freq=%d, schema=%d, passages=%s%s",
                fact.idx,
                fact.content,
                fact.frequency,
                fact.schema_idx,
                fact.passage_indices[:5],
                "..." if len(fact.passage_indices) > 5 else "",
            )

        logger.info("--- Passage Layer ---")
        for passage in self.passage_layer[:n]:
            content_preview = (
                passage.content[:80] + "..." if len(passage.content) > 80 else passage.content
            )
            logger.debug(
                "  [%d] %s modality=%s content=%s facts=%s%s",
                passage.idx,
                passage.chunk_id,
                passage.modality,
                content_preview,
                passage.fact_indices[:5],
                "..." if len(passage.fact_indices) > 5 else "",
            )


def load_openie_results(path: str) -> Dict[str, Any]:
    """Load OpenIE results JSON."""
    logger.info("Loading OpenIE results from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d docs", len(data.get("docs", [])))
    return data


def build_two_layer_memory_from_raw_openie(data: Dict[str, Any]) -> ThreeLayerMemory:
    """
    Build a ``ThreeLayerMemory`` with empty ``schema_layer`` from raw OpenIE ``data``
    (``docs`` with ``extracted_triples``, no ontology map).
    """
    memory = ThreeLayerMemory()
    memory.build_from_raw_openie_results(data)
    return memory


def main() -> None:
    input_path = DEFAULT_FILTERED_PATH
    output_path = DEFAULT_MEMORY_OUTPUT_PATH
    if not input_path or not output_path:
        raise SystemExit(
            "Set DEFAULT_FILTERED_PATH and DEFAULT_MEMORY_OUTPUT_PATH before running main()."
        )

    data = load_openie_results(input_path)

    memory = ThreeLayerMemory()
    memory.build_from_openie_results(data)

    memory.print_summary()
    memory.print_sample(n=3)

    memory.save(output_path)

    logger.info("--- Verification: reload and check ---")
    loaded_memory = ThreeLayerMemory.load(output_path)
    loaded_memory.print_summary()


if __name__ == "__main__":
    main()
