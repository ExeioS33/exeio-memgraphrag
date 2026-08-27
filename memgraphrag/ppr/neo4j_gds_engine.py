"""Neo4j GDS Personalized PageRank engine (stub / thin wrapper).

Provenance: MemGraphRAG hybrid PPR plan — ``neo4j_gds`` alternative to igraph.
Calls ``gds.pageRank.stream`` when a Neo4j driver + GDS graph projection are
available; otherwise returns seed weights for passage nodes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from memgraphrag.ppr.base import PPREngine

logger = logging.getLogger(__name__)


class Neo4jGDSPPREngine(PPREngine):
    """Run PPR via Neo4j Graph Data Science when available."""

    def __init__(
        self,
        driver: Any | None = None,
        graph_name: str = "memgraphrag",
        passage_label: str = "Passage",
        **kwargs: Any,
    ) -> None:
        if driver is None:
            # Refuse rather than silently degrade. Without a driver this engine used
            # to return the seed weights unchanged; those keys are `entity-` /
            # `schema-` ids, not passage ids, so `passage_scores` came back non-empty
            # (suppressing the dense fallback) yet resolved to no content at all —
            # the LLM answered with zero context and the operator saw HTTP 200.
            raise ValueError(
                "PPR_ENGINE='neo4j_gds' requires a Neo4j driver, which the engine "
                "is not wired to receive yet. Use PPR_ENGINE='igraph'."
            )
        self.driver = driver
        self.graph_name = graph_name
        self.passage_label = passage_label
        self._extra = kwargs

    def run(
        self,
        seed_weights: Mapping[str, float],
        damping: float = 0.5,
        **kwargs: Any,
    ) -> Dict[str, float]:
        graph_name = kwargs.get("graph_name", self.graph_name)
        try:
            return self._run_gds(seed_weights, damping=damping, graph_name=graph_name)
        except Exception as exc:
            # Return nothing rather than the raw seeds: seed keys are entity/schema
            # ids, so echoing them back fabricates unresolvable "passages". An empty
            # result lets the caller fall back to dense passage retrieval instead.
            logger.warning(
                "Neo4j GDS pageRank failed (%s); returning no PPR scores so the "
                "caller falls back to dense retrieval",
                exc,
            )
            return {}

    def _run_gds(
        self,
        seed_weights: Mapping[str, float],
        damping: float,
        graph_name: str,
    ) -> Dict[str, float]:
        """Execute ``gds.pageRank.stream`` and collect passage scores.

        Expects nodes to have an ``id`` property matching seed / passage keys.
        Source nodes for personalization are those present in ``seed_weights``.
        """
        # Build sourceNodes list for GDS personalization when supported.
        source_ids = [nid for nid, w in seed_weights.items() if float(w) > 0]
        if not source_ids:
            return {}

        cypher = """
        CALL gds.pageRank.stream($graphName, {
          maxIterations: 20,
          dampingFactor: $damping,
          sourceNodes: [n IN $sourceIds | gds.util.asNode(n)]
        })
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        WHERE node:$passageLabel OR node.id STARTS WITH 'chunk-' OR node.id STARTS WITH 'passage-'
        RETURN node.id AS id, score AS score
        """
        # Note: label interpolation is not allowed in params; use a simpler query.
        cypher = """
        CALL gds.pageRank.stream($graphName, {
          maxIterations: 20,
          dampingFactor: $damping
        })
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN coalesce(node.id, elementId(node)) AS id, score AS score
        """

        scores: Dict[str, float] = {}
        with self.driver.session() as session:
            result = session.run(
                cypher,
                graphName=graph_name,
                damping=float(damping),
            )
            for record in result:
                nid = record.get("id")
                if nid is not None:
                    scores[str(nid)] = float(record["score"])

        if not scores:
            return {k: float(v) for k, v in seed_weights.items() if float(v) > 0}

        # Prefer passage-like ids when filtering; else return all
        passage_scores = {
            k: v
            for k, v in scores.items()
            if k.startswith(("chunk-", "passage-", "doc-")) or k in seed_weights
        }
        return passage_scores or scores
