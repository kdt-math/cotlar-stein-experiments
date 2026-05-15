"""Greedy selection algorithms for Cotlar--Stein experiments."""

from collections.abc import Callable
from typing import Any

import numpy as np

from src.config import AlphaFunction
from src.cotlar import cotlar_matrix_norms, is_admissible, sum_operator_norm
from src.sampling import Field, Matrix

Sampler = Callable[..., Matrix] | Callable[..., tuple[Matrix, int]]


def _draw_candidate(
    sampler: Sampler,
    N: int,
    field: Field,
    rng: np.random.Generator,
) -> tuple[Matrix, int]:
    """Draw one candidate matrix.

    Some samplers return only a matrix. The rejection sampler returns
    `(matrix, attempts)`. This helper normalizes both cases.
    """
    sample = sampler(N=N, field=field, rng=rng)

    if isinstance(sample, tuple):
        matrix, attempts = sample
        return matrix, int(attempts)

    return sample, 1


def _final_stats(
    S_list: list[Matrix],
    draws: int,
    alpha: float,
    terminated: bool = True,
) -> dict[str, Any]:
    """Compute summary statistics for a selected family."""
    norm_A, norm_B = cotlar_matrix_norms(S_list)
    sum_norm = sum_operator_norm(S_list)
    cotlar_bound = float(np.sqrt(norm_A * norm_B))

    return {
        "accepted_count": len(S_list),
        "draws": draws,
        "terminated": terminated,
        "sum_norm": sum_norm,
        "A_norm": norm_A,
        "B_norm": norm_B,
        "alpha": alpha,
        "cotlar_bound": cotlar_bound,
    }


def run_version1(
    N: int,
    K: int,
    alpha_func: AlphaFunction,
    c: float,
    sampler: Sampler,
    field: Field = "complex",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Run Version 1: draw exactly K candidates.

    A candidate is accepted if appending it keeps both Cotlar--Stein matrix
    norms at most alpha(K, N, c). The final accepted family has size at most K.
    """
    if rng is None:
        rng = np.random.default_rng()

    alpha = alpha_func(K, N, c)
    selected: list[Matrix] = []
    total_draws = 0

    for _ in range(K):
        candidate, draw_cost = _draw_candidate(
            sampler=sampler,
            N=N,
            field=field,
            rng=rng,
        )
        total_draws += draw_cost

        proposed = [*selected, candidate]
        if is_admissible(proposed, alpha=alpha):
            selected.append(candidate)

    return _final_stats(
        S_list=selected,
        draws=total_draws,
        alpha=alpha,
        terminated=True,
    )


def run_version2(
    N: int,
    K: int,
    alpha_func: AlphaFunction,
    c: float,
    sampler: Sampler,
    field: Field = "complex",
    rng: np.random.Generator | None = None,
    max_draws: int = 100_000,
) -> dict[str, Any]:
    """Run Version 2: draw until K candidates are accepted.

    The algorithm terminates successfully if K matrices are accepted before
    max_draws proposals have been used.
    """
    if max_draws <= 0:
        msg = f"max_draws must be positive, got {max_draws}."
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    alpha = alpha_func(K, N, c)
    selected: list[Matrix] = []
    total_draws = 0

    while len(selected) < K and total_draws < max_draws:
        candidate, draw_cost = _draw_candidate(
            sampler=sampler,
            N=N,
            field=field,
            rng=rng,
        )
        total_draws += draw_cost

        if total_draws > max_draws:
            break

        proposed = [*selected, candidate]
        if is_admissible(proposed, alpha=alpha):
            selected.append(candidate)

    terminated = len(selected) == K

    return _final_stats(
        S_list=selected,
        draws=total_draws,
        alpha=alpha,
        terminated=terminated,
    )


def run_version3(
    N: int,
    K: int,
    alpha_func: AlphaFunction,
    c: float,
    sampler: Sampler,
    field: Field = "complex",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Run Version 3: draw exactly K candidates and accept all of them.

    This is the unfiltered baseline. Unlike Version 1 and Version 2, this
    version does not check whether appending a candidate keeps the Cotlar--Stein
    matrix norms below alpha(K, N, c). The final selected family always has
    size K.
    """
    if rng is None:
        rng = np.random.default_rng()

    alpha = alpha_func(K, N, c)
    selected: list[Matrix] = []
    total_draws = 0

    for _ in range(K):
        candidate, draw_cost = _draw_candidate(
            sampler=sampler,
            N=N,
            field=field,
            rng=rng,
        )
        total_draws += draw_cost
        selected.append(candidate)

    return _final_stats(
        S_list=selected,
        draws=total_draws,
        alpha=alpha,
        terminated=True,
    )
