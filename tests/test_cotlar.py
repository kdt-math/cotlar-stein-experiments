"""Tests for Cotlar--Stein matrix utilities."""

import numpy as np
import pytest

from src.cotlar import (
    adjoint,
    build_A,
    build_B,
    cotlar_bound,
    cotlar_matrix_norms,
    is_admissible,
    sum_operator,
    sum_operator_norm,
)


def test_adjoint_real_matrix() -> None:
    T = np.array([[1.0, 2.0], [3.0, 4.0]])

    expected = np.array([[1.0, 3.0], [2.0, 4.0]])

    assert np.allclose(adjoint(T), expected)


def test_adjoint_complex_matrix() -> None:
    T = np.array([[1.0 + 1.0j, 2.0], [3.0j, 4.0]])

    expected = np.array([[1.0 - 1.0j, -3.0j], [2.0, 4.0]])

    assert np.allclose(adjoint(T), expected)


def test_empty_family() -> None:
    assert build_A([]).shape == (0, 0)
    assert build_B([]).shape == (0, 0)
    assert cotlar_matrix_norms([]) == (0.0, 0.0)
    assert sum_operator([]) is None
    assert sum_operator_norm([]) == 0.0
    assert cotlar_bound([]) == 0.0
    assert is_admissible([], alpha=0.0)


def test_single_identity() -> None:
    identity = np.eye(2)
    S_list = [identity]

    assert np.allclose(build_A(S_list), np.array([[1.0]]))
    assert np.allclose(build_B(S_list), np.array([[1.0]]))
    assert cotlar_matrix_norms(S_list) == (1.0, 1.0)
    assert sum_operator_norm(S_list) == 1.0
    assert cotlar_bound(S_list) == 1.0
    assert is_admissible(S_list, alpha=1.0)
    assert not is_admissible(S_list, alpha=0.99)


def test_identity_and_zero() -> None:
    identity = np.eye(2)
    Z = np.zeros((2, 2))
    S_list = [identity, Z]

    expected = np.array([[1.0, 0.0], [0.0, 0.0]])

    assert np.allclose(build_A(S_list), expected)
    assert np.allclose(build_B(S_list), expected)
    assert cotlar_matrix_norms(S_list) == (1.0, 1.0)
    assert sum_operator_norm(S_list) == 1.0


def test_two_identical_identities() -> None:
    identity = np.eye(2)
    S_list = [identity, identity]

    expected = np.ones((2, 2))

    assert np.allclose(build_A(S_list), expected)
    assert np.allclose(build_B(S_list), expected)
    assert cotlar_matrix_norms(S_list) == (2.0, 2.0)
    assert sum_operator_norm(S_list) == 2.0
    assert cotlar_bound(S_list) == 2.0


def test_orthogonal_projections() -> None:
    P1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    P2 = np.array([[0.0, 0.0], [0.0, 1.0]])
    S_list = [P1, P2]

    expected = np.eye(2)

    assert np.allclose(build_A(S_list), expected)
    assert np.allclose(build_B(S_list), expected)
    assert cotlar_matrix_norms(S_list) == (1.0, 1.0)
    assert sum_operator_norm(S_list) == 1.0
    assert cotlar_bound(S_list) == 1.0


def test_is_admissible_rejects_negative_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be nonnegative"):
        is_admissible([], alpha=-1.0)
