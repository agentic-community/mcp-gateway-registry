"""Tests for the cosine_similarity helper shared between dedup and search."""

import math

import pytest

from registry.utils.vector import cosine_similarity


class TestCosineSimilarity:
    def test_identical_unit_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_unit_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposing_vectors(self) -> None:
        # Cosine of opposite-direction vectors is -1.
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0

    def test_known_angle(self) -> None:
        # 45 degrees between [1, 0] and [1, 1]/sqrt(2) -> cosine ~0.7071.
        norm = math.sqrt(2)
        result = cosine_similarity([1.0, 0.0], [1.0 / norm, 1.0 / norm])
        assert math.isclose(result, 1.0 / norm, rel_tol=1e-9)

    @pytest.mark.parametrize(
        "a,b",
        [
            ([], [1.0]),
            ([1.0], []),
            ([], []),
        ],
    )
    def test_empty_vector_returns_zero(self, a: list[float], b: list[float]) -> None:
        assert cosine_similarity(a, b) == 0.0

    def test_length_mismatch_returns_zero(self) -> None:
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_zero_magnitude_first_vector_returns_zero(self) -> None:
        # All-zero vector has magnitude 0; cosine is undefined.
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_magnitude_second_vector_returns_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0
