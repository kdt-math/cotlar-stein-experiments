"""Tests for greedy Cotlar--Stein selection algorithms."""

import numpy as np
import pytest

from src.config import alpha_constant, alpha_linear
from src.greedy import run_version1, run_version2, run_version3


def zero_sampler(
    N: int,
    field: str = "complex",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Always return the zero matrix."""
    dtype = complex if field == "complex" else float
    return np.zeros((N, N), dtype=dtype)


def identity_sampler(
    N: int,
    field: str = "complex",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Always return the identity matrix."""
    dtype = complex if field == "complex" else float
    return np.eye(N, dtype=dtype)


def test_version1_accepts_all_zero_matrices() -> None:
    result = run_version1(
        N=3,
        K=5,
        alpha_func=alpha_constant,
        c=0.0,
        sampler=zero_sampler,
        field="complex",
        rng=np.random.default_rng(123),
    )

    assert result["accepted_count"] == 5
    assert result["draws"] == 5
    assert result["terminated"]
    assert result["sum_norm"] == 0.0
    assert result["A_norm"] == 0.0
    assert result["B_norm"] == 0.0
    assert result["cotlar_bound"] == 0.0


def test_version1_with_identities_and_linear_alpha() -> None:
    result = run_version1(
        N=3,
        K=5,
        alpha_func=alpha_linear,
        c=1.0,
        sampler=identity_sampler,
        field="real",
        rng=np.random.default_rng(123),
    )

    assert result["accepted_count"] == 5
    assert result["draws"] == 5
    assert result["terminated"]
    assert np.isclose(result["sum_norm"], 5.0)
    assert np.isclose(result["A_norm"], 5.0)
    assert np.isclose(result["B_norm"], 5.0)
    assert np.isclose(result["alpha"], 5.0)
    assert np.isclose(result["cotlar_bound"], 5.0)


def test_version1_rejects_second_identity_when_alpha_too_small() -> None:
    result = run_version1(
        N=3,
        K=5,
        alpha_func=alpha_constant,
        c=1.0,
        sampler=identity_sampler,
        field="real",
        rng=np.random.default_rng(123),
    )

    assert result["accepted_count"] == 1
    assert result["draws"] == 5
    assert result["terminated"]
    assert result["sum_norm"] == 1.0
    assert result["A_norm"] == 1.0
    assert result["B_norm"] == 1.0
    assert result["cotlar_bound"] == 1.0


def test_version2_accepts_k_zero_matrices() -> None:
    result = run_version2(
        N=3,
        K=5,
        alpha_func=alpha_constant,
        c=0.0,
        sampler=zero_sampler,
        field="complex",
        rng=np.random.default_rng(123),
        max_draws=100,
    )

    assert result["accepted_count"] == 5
    assert result["draws"] == 5
    assert result["terminated"]
    assert result["sum_norm"] == 0.0
    assert result["cotlar_bound"] == 0.0


def test_version2_can_fail_to_terminate() -> None:
    result = run_version2(
        N=3,
        K=2,
        alpha_func=alpha_constant,
        c=0.5,
        sampler=identity_sampler,
        field="real",
        rng=np.random.default_rng(123),
        max_draws=10,
    )

    assert result["accepted_count"] == 0
    assert result["draws"] == 10
    assert not result["terminated"]
    assert result["sum_norm"] == 0.0
    assert result["cotlar_bound"] == 0.0


def test_version2_rejects_nonpositive_max_draws() -> None:
    with pytest.raises(ValueError, match="max_draws must be positive"):
        run_version2(
            N=3,
            K=2,
            alpha_func=alpha_constant,
            c=1.0,
            sampler=identity_sampler,
            max_draws=0,
        )


def test_version3_accepts_all_zero_matrices() -> None:
    result = run_version3(
        N=3,
        K=5,
        alpha_func=alpha_constant,
        c=0.0,
        sampler=zero_sampler,
        field="complex",
        rng=np.random.default_rng(123),
    )

    assert result["accepted_count"] == 5
    assert result["draws"] == 5
    assert result["terminated"]
    assert result["sum_norm"] == 0.0
    assert result["A_norm"] == 0.0
    assert result["B_norm"] == 0.0
    assert result["alpha"] == 0.0
    assert result["cotlar_bound"] == 0.0


def test_version3_accepts_all_identities_even_when_alpha_too_small() -> None:
    result = run_version3(
        N=3,
        K=5,
        alpha_func=alpha_constant,
        c=1.0,
        sampler=identity_sampler,
        field="real",
        rng=np.random.default_rng(123),
    )

    assert result["accepted_count"] == 5
    assert result["draws"] == 5
    assert result["terminated"]
    assert np.isclose(result["sum_norm"], 5.0)
    assert np.isclose(result["A_norm"], 5.0)
    assert np.isclose(result["B_norm"], 5.0)
    assert np.isclose(result["alpha"], 1.0)
    assert np.isclose(result["cotlar_bound"], 5.0)


def test_version3_with_identities_and_linear_alpha() -> None:
    result = run_version3(
        N=3,
        K=5,
        alpha_func=alpha_linear,
        c=1.0,
        sampler=identity_sampler,
        field="real",
        rng=np.random.default_rng(123),
    )

    assert result["accepted_count"] == 5
    assert result["draws"] == 5
    assert result["terminated"]
    assert np.isclose(result["sum_norm"], 5.0)
    assert np.isclose(result["A_norm"], 5.0)
    assert np.isclose(result["B_norm"], 5.0)
    assert np.isclose(result["alpha"], 5.0)
    assert np.isclose(result["cotlar_bound"], 5.0)