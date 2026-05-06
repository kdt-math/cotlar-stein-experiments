"""Cotlar--Stein matrix utilities."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.sampling import Matrix, spectral_norm

ScalarMatrix = NDArray[np.float64]


def adjoint(T: Matrix) -> Matrix:
    """Return the adjoint, i.e. conjugate transpose, of a matrix."""
    return T.conj().T


def build_A(S_list: Sequence[Matrix]) -> ScalarMatrix:
    """Build the Cotlar--Stein matrix A.

    For S_list = [S_1, ..., S_r], this returns the r by r scalar matrix

        A[j, k] = sqrt(||S_j S_k^*||_2).

    The indices j and k refer to operators in the selected family, not to
    coordinates in the Hilbert space.
    """
    r = len(S_list)
    A = np.zeros((r, r), dtype=float)

    for j, S_j in enumerate(S_list):
        for k, S_k in enumerate(S_list):
            A[j, k] = np.sqrt(spectral_norm(S_j @ adjoint(S_k)))

    return A


def build_B(S_list: Sequence[Matrix]) -> ScalarMatrix:
    """Build the Cotlar--Stein matrix B.

    For S_list = [S_1, ..., S_r], this returns the r by r scalar matrix

        B[j, k] = sqrt(||S_j^* S_k||_2).

    The indices j and k refer to operators in the selected family, not to
    coordinates in the Hilbert space.
    """
    r = len(S_list)
    B = np.zeros((r, r), dtype=float)

    for j, S_j in enumerate(S_list):
        for k, S_k in enumerate(S_list):
            B[j, k] = np.sqrt(spectral_norm(adjoint(S_j) @ S_k))

    return B


def cotlar_matrix_norms(S_list: Sequence[Matrix]) -> tuple[float, float]:
    """Return the spectral norms of A_S and B_S.

    For the empty family, both norms are defined to be 0.
    """
    if len(S_list) == 0:
        return 0.0, 0.0

    A = build_A(S_list)
    B = build_B(S_list)

    return spectral_norm(A), spectral_norm(B)


def sum_operator(S_list: Sequence[Matrix]) -> Matrix | None:
    """Return the operator sum of the selected family.

    For the empty family, return None. Use `sum_operator_norm` if you only need
    the norm.
    """
    if len(S_list) == 0:
        return None

    total = np.zeros_like(S_list[0])
    for S in S_list:
        total = total + S

    return total


def sum_operator_norm(S_list: Sequence[Matrix]) -> float:
    """Return ||sum_j S_j||_2.

    For the empty family, this is defined to be 0.
    """
    total = sum_operator(S_list)

    if total is None:
        return 0.0

    return spectral_norm(total)


def cotlar_bound(S_list: Sequence[Matrix]) -> float:
    """Return sqrt(||A_S||_2 ||B_S||_2)."""
    norm_A, norm_B = cotlar_matrix_norms(S_list)
    return float(np.sqrt(norm_A * norm_B))


def is_admissible(
    S_list: Sequence[Matrix],
    alpha: float,
    tol: float = 1e-12,
) -> bool:
    """Return True iff both Cotlar--Stein matrix norms are at most alpha.

    The tolerance avoids rejecting borderline admissible families because of
    floating-point roundoff.
    """
    if alpha < 0:
        msg = f"alpha must be nonnegative, got {alpha}."
        raise ValueError(msg)
    if tol < 0:
        msg = f"tol must be nonnegative, got {tol}."
        raise ValueError(msg)

    norm_A, norm_B = cotlar_matrix_norms(S_list)
    return norm_A <= alpha + tol and norm_B <= alpha + tol
