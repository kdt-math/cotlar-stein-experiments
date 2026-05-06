"""Configuration and threshold functions for Cotlar--Stein experiments."""

from collections.abc import Callable
from math import sqrt

AlphaFunction = Callable[[int, int, float], float]


def validate_parameters(K: int, N: int, c: float) -> None:
    """Validate common parameters for threshold functions.

    Args:
        K: Number of draws or target accepted operators.
        N: Dimension of the Hilbert space.
        c: Threshold parameter.

    Raises:
        ValueError: If one of the parameters is outside its expected range.
    """
    if K <= 0:
        msg = f"K must be positive, got {K}."
        raise ValueError(msg)
    if N <= 0:
        msg = f"N must be positive, got {N}."
        raise ValueError(msg)
    if c < 0:
        msg = f"c must be nonnegative, got {c}."
        raise ValueError(msg)


def alpha_linear(K: int, N: int, c: float) -> float:
    """Linear threshold alpha(K, N, c) = 1 + c(K - 1)."""
    validate_parameters(K, N, c)
    return 1.0 + c * (K - 1)


def alpha_sqrt(K: int, N: int, c: float) -> float:
    """Square-root threshold alpha(K, N, c) = 1 + c sqrt(K)."""
    validate_parameters(K, N, c)
    return 1.0 + c * (sqrt(K) - 1)


def alpha_constant(K: int, N: int, c: float) -> float:
    """Constant threshold alpha(K, N, c) = c.

    Here c itself is the threshold. For example, c=2 gives alpha=2.
    """
    validate_parameters(K, N, c)
    return float(c)


ALPHA_FUNCTIONS: dict[str, AlphaFunction] = {
    "linear": alpha_linear,
    "sqrt": alpha_sqrt,
    "constant": alpha_constant,
}


def get_alpha_function(name: str) -> AlphaFunction:
    """Return a threshold function by name."""
    try:
        return ALPHA_FUNCTIONS[name]
    except KeyError as exc:
        available = ", ".join(sorted(ALPHA_FUNCTIONS))
        msg = f"Unknown alpha function {name!r}. Available options: {available}."
        raise ValueError(msg) from exc


DEFAULT_N_VALUES = [2, 3, 4]
DEFAULT_K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
DEFAULT_C_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
DEFAULT_TRIALS = 100
DEFAULT_FIELD = "complex"
DEFAULT_FIELD_VALUES = ["real", "complex"]
DEFAULT_ALPHA_NAME = "linear"