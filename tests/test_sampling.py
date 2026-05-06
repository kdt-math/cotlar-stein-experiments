"""Tests for random matrix samplers."""

import numpy as np
import pytest

from src.sampling import (
    gaussian_matrix,
    real_dimension,
    sample_frobenius_ball,
    sample_spectral_ball_rejection,
    sample_spectral_ball_scaled_gaussian,
    spectral_norm,
)


def test_real_dimension() -> None:
    assert real_dimension(N=3, field="real") == 9
    assert real_dimension(N=3, field="complex") == 18


def test_real_dimension_rejects_invalid_field() -> None:
    with pytest.raises(ValueError, match="field must be"):
        real_dimension(N=3, field="quaternion")  # type: ignore[arg-type]


def test_gaussian_matrix_real() -> None:
    rng = np.random.default_rng(123)
    T = gaussian_matrix(N=4, field="real", rng=rng)

    assert T.shape == (4, 4)
    assert np.isrealobj(T)


def test_gaussian_matrix_complex() -> None:
    rng = np.random.default_rng(123)
    T = gaussian_matrix(N=4, field="complex", rng=rng)

    assert T.shape == (4, 4)
    assert np.iscomplexobj(T)


@pytest.mark.parametrize("field", ["real", "complex"])
def test_sample_frobenius_ball_has_correct_shape_and_bound(field: str) -> None:
    rng = np.random.default_rng(123)
    radius = 2.0

    T = sample_frobenius_ball(N=3, field=field, radius=radius, rng=rng)

    assert T.shape == (3, 3)
    assert np.linalg.norm(T, ord="fro") <= radius + 1e-12


@pytest.mark.parametrize("field", ["real", "complex"])
def test_sample_spectral_ball_rejection_has_spectral_norm_at_most_one(
    field: str,
) -> None:
    rng = np.random.default_rng(123)

    T, attempts = sample_spectral_ball_rejection(
        N=2,
        field=field,
        max_attempts=10_000,
        rng=rng,
    )

    assert T.shape == (2, 2)
    assert attempts >= 1
    assert spectral_norm(T) <= 1.0 + 1e-12


@pytest.mark.parametrize("field", ["real", "complex"])
def test_sample_spectral_ball_scaled_gaussian_has_spectral_norm_at_most_one(
    field: str,
) -> None:
    rng = np.random.default_rng(123)

    T = sample_spectral_ball_scaled_gaussian(N=5, field=field, rng=rng)

    assert T.shape == (5, 5)
    assert spectral_norm(T) <= 1.0 + 1e-12


def test_scaled_gaussian_is_reproducible_with_same_seed() -> None:
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    T1 = sample_spectral_ball_scaled_gaussian(N=3, field="complex", rng=rng1)
    T2 = sample_spectral_ball_scaled_gaussian(N=3, field="complex", rng=rng2)

    assert np.allclose(T1, T2)
