"""Random matrix samplers for Cotlar--Stein experiments."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

Field = Literal["real", "complex"]
Matrix = NDArray[np.float64] | NDArray[np.complex128]


def validate_dimension(N: int) -> None:
    """Validate the matrix dimension."""
    if N <= 0:
        msg = f"N must be positive, got {N}."
        raise ValueError(msg)


def validate_field(field: str) -> None:
    """Validate the scalar field."""
    if field not in {"real", "complex"}:
        msg = f"field must be either 'real' or 'complex', got {field!r}."
        raise ValueError(msg)


def real_dimension(N: int, field: Field = "complex") -> int:
    """Return the real dimension of the space of N by N matrices."""
    validate_dimension(N)
    validate_field(field)

    if field == "real":
        return N * N
    return 2 * N * N


def gaussian_matrix(
    N: int,
    field: Field = "complex",
    rng: np.random.Generator | None = None,
) -> Matrix:
    """Sample a real or complex Gaussian N by N matrix.

    For the complex case, the real and imaginary parts are sampled independently
    from the standard normal distribution.
    """
    validate_dimension(N)
    validate_field(field)

    if rng is None:
        rng = np.random.default_rng()

    if field == "real":
        return rng.normal(size=(N, N))

    real_part = rng.normal(size=(N, N))
    imag_part = rng.normal(size=(N, N))
    return real_part + 1j * imag_part


def sample_frobenius_ball(
    N: int,
    field: Field = "complex",
    radius: float | None = None,
    rng: np.random.Generator | None = None,
) -> Matrix:
    """Sample uniformly from a Frobenius norm ball.

    The ambient real dimension is N^2 for real matrices and 2N^2 for complex
    matrices. If radius is None, we use radius sqrt(N), which contains the
    spectral norm unit ball.
    """
    validate_dimension(N)
    validate_field(field)

    if radius is None:
        radius = float(np.sqrt(N))

    if radius <= 0:
        msg = f"radius must be positive, got {radius}."
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    direction = gaussian_matrix(N=N, field=field, rng=rng)
    direction_norm = np.linalg.norm(direction, ord="fro")

    # This is overwhelmingly unlikely, but avoids division by zero.
    while direction_norm == 0:
        direction = gaussian_matrix(N=N, field=field, rng=rng)
        direction_norm = np.linalg.norm(direction, ord="fro")

    direction = direction / direction_norm

    dim = real_dimension(N=N, field=field)
    radial_factor = rng.random() ** (1.0 / dim)

    return radius * radial_factor * direction


def spectral_norm(T: Matrix) -> float:
    """Return the induced 2-norm of a matrix."""
    return float(np.linalg.norm(T, ord=2))


def sample_spectral_ball_rejection(
    N: int,
    field: Field = "complex",
    max_attempts: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[Matrix, int]:
    """Sample uniformly from the spectral norm unit ball by rejection.

    This is exact with respect to Lebesgue measure on the spectral norm ball,
    because we sample uniformly from a containing Frobenius ball and reject
    points outside the spectral norm ball.

    This method is intended only for small N. It may be slow in higher
    dimensions.

    Returns:
        A pair `(T, attempts)`, where T is the accepted matrix and attempts is
        the number of proposals used.
    """
    validate_dimension(N)
    validate_field(field)

    if max_attempts <= 0:
        msg = f"max_attempts must be positive, got {max_attempts}."
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    for attempt in range(1, max_attempts + 1):
        T = sample_frobenius_ball(N=N, field=field, radius=np.sqrt(N), rng=rng)
        if spectral_norm(T) <= 1.0:
            return T, attempt

    msg = (
        "Rejection sampler failed to accept a matrix after "
        f"{max_attempts} attempts. Try smaller N or larger max_attempts."
    )
    raise RuntimeError(msg)


def sample_spectral_ball_scaled_gaussian(
    N: int,
    field: Field = "complex",
    rng: np.random.Generator | None = None,
) -> Matrix:
    """Fast surrogate sampler for the spectral norm unit ball.

    This is not uniform from the spectral norm ball.

    We sample a Gaussian matrix, normalize it to have spectral norm 1, and then
    multiply by a random radial factor. The output always has spectral norm at
    most 1, but its distribution is only a convenient surrogate.
    """
    validate_dimension(N)
    validate_field(field)

    if rng is None:
        rng = np.random.default_rng()

    T = gaussian_matrix(N=N, field=field, rng=rng)
    norm_T = spectral_norm(T)

    while norm_T == 0:
        T = gaussian_matrix(N=N, field=field, rng=rng)
        norm_T = spectral_norm(T)

    dim = real_dimension(N=N, field=field)
    radial_factor = rng.random() ** (1.0 / dim)

    return radial_factor * T / norm_T
