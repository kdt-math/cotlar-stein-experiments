"""Tests for threshold functions."""

import pytest

from src.config import (
    DEFAULT_ALPHA_NAME,
    DEFAULT_C_VALUES,
    DEFAULT_FIELD,
    DEFAULT_FIELD_VALUES,
    DEFAULT_K_VALUES,
    DEFAULT_N_VALUES,
    DEFAULT_TRIALS,
    alpha_constant,
    alpha_linear,
    alpha_sqrt,
    get_alpha_function,
)


def test_alpha_linear() -> None:
    assert alpha_linear(K=5, N=3, c=0.5) == 3.0


def test_alpha_sqrt() -> None:
    assert alpha_sqrt(K=4, N=3, c=2.0) == 3.0


def test_alpha_constant() -> None:
    assert alpha_constant(K=10, N=3, c=2.5) == 2.5


def test_get_alpha_function() -> None:
    alpha = get_alpha_function("linear")
    assert alpha(K=5, N=3, c=0.5) == 3.0


def test_get_alpha_function_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown alpha function"):
        get_alpha_function("not-a-threshold")


@pytest.mark.parametrize(
    ("K", "N", "c"),
    [
        (0, 3, 0.5),
        (5, 0, 0.5),
        (5, 3, -0.5),
    ],
)
def test_alpha_functions_validate_parameters(K: int, N: int, c: float) -> None:
    with pytest.raises(ValueError):
        alpha_linear(K=K, N=N, c=c)


def test_defaults_are_nonempty() -> None:
    assert DEFAULT_N_VALUES
    assert DEFAULT_K_VALUES
    assert DEFAULT_C_VALUES
    assert DEFAULT_TRIALS > 0
    assert DEFAULT_ALPHA_NAME in {"linear", "sqrt", "constant"}
    assert set(DEFAULT_FIELD_VALUES) == {"real", "complex"}
    assert DEFAULT_FIELD in DEFAULT_FIELD_VALUES
