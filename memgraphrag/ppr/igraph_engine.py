"""igraph-backed Personalized PageRank engine.

Provenance: adapted from MemGraphRAG ``code/src/MemGraphRAG.py`` ``run_ppr``
(``igraph.Graph.personalized_pagerank`` with ``prpack``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, Sequence

from memgraphrag.ppr.base import PPREngine

logger = logging.getLogger(__name__)


class IgraphPPREngine(PPREngine):
    """Run PPR on an in-memory igraph Graph."""

    def __init__(
        self,
        edges: Sequence[tuple[str, str]] | None = None,
        edge_weights: Sequence[float] | None = None,
        passage_ids: Iterable[str] | None = None,
        directed: bool = False,
        graph: Any | None = None,
    ) -> None:
        """Build or wrap an igraph graph.

        Args:
            edges: Undirected/directed edge list of (source, target) node ids.
            edge_weights: Optional per-edge weights aligned with ``edges``.
            passage_ids: Node ids treated as passage nodes in the score output.
            directed: Whether the graph is directed.
            graph: Optional pre-built ``igraph.Graph`` (skips edge construction).
        """
        try:
            import igraph as ig
        except ImportError as exc:  # pragma: no cover
            raise ImportError("python-igraph is required for IgraphPPREngine") from exc

        self._ig = ig
        self.passage_ids = set(passage_ids or [])

        if graph is not None:
            self.graph = graph
        else:
            edges = list(edges or [])
            nodes: list[str] = []
            seen: set[str] = set()
            for s, t in edges:
                if s not in seen:
                    nodes.append(s)
                    seen.add(s)
                if t not in seen:
                    nodes.append(t)
                    seen.add(t)
            # Include isolated passage seeds if any
            for pid in self.passage_ids:
                if pid not in seen:
                    nodes.append(pid)
                    seen.add(pid)

            g = ig.Graph(directed=directed)
            g.add_vertices(nodes)
            if edges:
                g.add_edges(edges)
            if edge_weights is not None and len(edge_weights) == len(edges):
                g.es["weight"] = list(edge_weights)
            else:
                g.es["weight"] = [1.0] * len(edges)
            self.graph = g

        # Cache name → vertex index
        names = self.graph.vs["name"] if "name" in self.graph.vs.attributes() else None
        if names is None:
            # vertices were added as strings → name attribute set automatically
            names = [str(v.index) for v in self.graph.vs]
            self.graph.vs["name"] = names
        self.name_to_idx: Dict[str, int] = {str(n): i for i, n in enumerate(names)}

        if not self.passage_ids:
            # Heuristic: ids prefixed with "chunk-" / "passage-" are passages
            self.passage_ids = {
                n for n in self.name_to_idx if n.startswith(("chunk-", "passage-", "doc-"))
            }

    def set_passage_ids(self, passage_ids: Iterable[str]) -> None:
        self.passage_ids = set(passage_ids)

    def run(
        self,
        seed_weights: Mapping[str, float],
        damping: float = 0.5,
        **kwargs: Any,
    ) -> Dict[str, float]:
        n = self.graph.vcount()
        if n == 0:
            return {}

        reset = [0.0] * n
        for name, weight in seed_weights.items():
            idx = self.name_to_idx.get(str(name))
            if idx is not None and weight is not None and weight > 0:
                reset[idx] = float(weight)

        total = sum(reset)
        if total <= 0:
            # Uniform teleport if no valid seeds
            reset = [1.0 / n] * n
        else:
            reset = [w / total for w in reset]

        use_weights = "weight" in self.graph.es.attributes()
        try:
            scores = self.graph.personalized_pagerank(
                vertices=range(n),
                damping=float(damping),
                directed=self.graph.is_directed(),
                weights="weight" if use_weights else None,
                reset=reset,
                implementation=kwargs.get("implementation", "prpack"),
            )
        except Exception as exc:
            logger.warning("igraph PPR failed (%s); falling back to seed scores", exc)
            return {
                name: float(seed_weights.get(name, 0.0))
                for name in (self.passage_ids or self.name_to_idx)
            }

        passage_scores: Dict[str, float] = {}
        targets = self.passage_ids or set(self.name_to_idx.keys())
        for name in targets:
            idx = self.name_to_idx.get(name)
            if idx is not None:
                passage_scores[name] = float(scores[idx])
        return passage_scores
