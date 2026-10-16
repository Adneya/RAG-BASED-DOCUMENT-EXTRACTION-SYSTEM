from __future__ import annotations

import hashlib
import math
import re


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}")


class HashingEmbeddingFunction:
    """Small deterministic embedding function for offline Chroma retrieval."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = max(32, dimensions)

    @staticmethod
    def name() -> str:
        return "hashing_embedding_function"

    @staticmethod
    def is_legacy() -> bool:
        return True

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in input]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokens(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
        words = TOKEN_PATTERN.findall(text.lower())
        if len(words) < 2:
            return words

        bigrams = [f"{left}_{right}" for left, right in zip(words, words[1:])]
        return words + bigrams
