"""Stable word shingles and universal-hash MinHash signatures."""

import hashlib
import re
from collections.abc import Iterable, Sequence

import numpy as np

PRIME = np.uint64(2_147_483_647)
EMPTY_SIGNATURE = np.uint64(2_147_483_647)
_TOKEN = re.compile(r"[a-z0-9]+")


def word_shingles(text: str, size: int = 3) -> np.ndarray:
    """Return sorted, unique 32-bit hashes of lower-cased word shingles."""
    if size < 1:
        raise ValueError("shingle size must be positive")
    words = _TOKEN.findall(text.lower())
    if not words:
        return np.empty(0, dtype=np.uint32)
    if len(words) < size:
        shingles = [" ".join(words)]
    else:
        shingles = [" ".join(words[start : start + size]) for start in range(len(words) - size + 1)]
    values = {
        int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=4).digest(), "little")
        & 0x7FFF_FFFF
        for item in shingles
    }
    return np.array(sorted(values), dtype=np.uint32)


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Compute exact Jaccard similarity for two sorted shingle arrays."""
    if left.size == 0 and right.size == 0:
        return 1.0
    intersection = np.intersect1d(left, right, assume_unique=True).size
    union = left.size + right.size - intersection
    return intersection / union if union else 0.0


class MinHash:
    """MinHash using independent affine hashes modulo a Mersenne prime."""

    def __init__(self, num_permutations: int = 128, seed: int = 13) -> None:
        if num_permutations < 1:
            raise ValueError("num_permutations must be positive")
        self.num_permutations = num_permutations
        self.seed = seed
        generator = np.random.default_rng(seed)
        self.multipliers = generator.integers(
            1, int(PRIME), size=num_permutations, dtype=np.uint64
        )
        self.offsets = generator.integers(
            0, int(PRIME), size=num_permutations, dtype=np.uint64
        )

    def signature(self, shingles: Iterable[int] | np.ndarray) -> np.ndarray:
        """Compute one minimum per permutation; empty sets share a sentinel."""
        values = np.asarray(list(shingles) if not isinstance(shingles, np.ndarray) else shingles)
        if values.size == 0:
            return np.full(self.num_permutations, EMPTY_SIGNATURE, dtype=np.uint64)
        values = values.astype(np.uint64, copy=False).reshape(-1)
        if np.any(values > PRIME):
            raise ValueError("shingle hashes must fit in the 31-bit field")
        hashed = (values[:, None] * self.multipliers[None, :] + self.offsets[None, :]) % PRIME
        return hashed.min(axis=0)

    def transform(self, texts: Sequence[str], shingle_size: int = 3) -> np.ndarray:
        return np.stack(
            [self.signature(word_shingles(text, shingle_size)) for text in texts]
        ).astype(np.uint64, copy=False)

    @staticmethod
    def estimate_jaccard(left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape or left.ndim != 1 or left.size == 0:
            raise ValueError("signatures must be non-empty vectors of equal length")
        return float(np.mean(left == right))
