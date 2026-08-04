"""Abstract Personalized PageRank engine interface.

Provenance: abstracted from MemGraphRAG ``code/src/MemGraphRAG.py`` ``run_ppr``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class PPREngine(ABC):
    """Run personalized PageRank and return passage-level scores."""

    @abstractmethod
    def run(
        self,
        seed_weights: Dict[str, float],
        damping: float = 0.5,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute passage scores from seed node weights.

        Args:
            seed_weights: Mapping of node id → personalization / reset weight.
            damping: Teleport damping factor (typically 0.5 in MemGraphRAG).

        Returns:
            Mapping of passage node id → PPR score (unsorted).
        """
