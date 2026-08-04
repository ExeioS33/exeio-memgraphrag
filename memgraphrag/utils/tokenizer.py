"""Tiktoken-based tokenizer for MemGraphRAG.

Adapted from LightRAG ``lightrag/utils.py`` (``TiktokenTokenizer`` / ``Tokenizer``).
"""

from __future__ import annotations

from typing import List


class TiktokenTokenizer:
    """Thin wrapper around ``tiktoken`` with encode / decode / encode_batch."""

    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        """Initialize a tokenizer for ``model_name``.

        Raises:
            ImportError: If tiktoken is not installed.
            ValueError: If ``model_name`` is not recognized by tiktoken.
        """
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "tiktoken is not installed. Install it with `pip install tiktoken`."
            ) from exc

        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError as exc:
            raise ValueError(f"Invalid model_name: {model_name}.") from exc

        self.model_name = model_name

    def encode(self, content: str) -> List[int]:
        """Encode ``content`` into token ids."""
        try:
            return self._encoding.encode(content)
        except ValueError as exc:
            # Allow special-token strings that appear in user corpora.
            if "special token" not in str(exc):
                raise
            return self._encoding.encode(content, disallowed_special=())

    def decode(self, tokens: List[int]) -> str:
        """Decode token ids back into a string."""
        return self._encoding.decode(tokens)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of strings into lists of token ids."""
        try:
            return self._encoding.encode_batch(texts)
        except ValueError:
            return [self.encode(text) for text in texts]
