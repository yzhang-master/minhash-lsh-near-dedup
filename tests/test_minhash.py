import numpy as np
import pytest

from minhash_lsh.minhash import MinHash, jaccard, word_shingles


def test_word_shingles_are_stable_unique_and_ordered() -> None:
    first = word_shingles("One two, two three ONE four", size=2)
    second = word_shingles("one TWO two three one FOUR", size=2)

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint32
    assert len(first) == len(set(first.tolist()))


def test_minhash_estimate_tracks_exact_jaccard() -> None:
    left = "alpha beta gamma delta epsilon zeta eta theta"
    right = "alpha beta gamma delta epsilon zeta eta different"
    left_shingles = word_shingles(left, size=1)
    right_shingles = word_shingles(right, size=1)
    signature = MinHash(num_permutations=256, seed=7)

    estimate = signature.estimate_jaccard(
        signature.signature(left_shingles), signature.signature(right_shingles)
    )

    assert estimate == pytest.approx(jaccard(left_shingles, right_shingles), abs=0.08)


def test_identical_and_empty_documents_have_well_defined_similarity() -> None:
    signature = MinHash(num_permutations=32)
    empty = word_shingles("!!!")
    text = word_shingles("a b c", size=2)

    assert signature.estimate_jaccard(signature.signature(empty), signature.signature(empty)) == 1.0
    assert jaccard(empty, empty) == 1.0
    assert jaccard(empty, text) == 0.0
    np.testing.assert_array_equal(signature.signature(text), signature.signature(text))


def test_seeds_change_signatures_but_not_reproducibility() -> None:
    shingles = word_shingles("one two three four", size=1)
    first = MinHash(64, seed=1).signature(shingles)
    repeated = MinHash(64, seed=1).signature(shingles)
    different = MinHash(64, seed=2).signature(shingles)

    np.testing.assert_array_equal(first, repeated)
    assert np.any(first != different)


def test_invalid_configuration_and_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        MinHash(0)
    with pytest.raises(ValueError, match="positive"):
        word_shingles("text", size=0)
    with pytest.raises(ValueError, match="equal length"):
        MinHash.estimate_jaccard(np.ones(2), np.ones(3))
