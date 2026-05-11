"""Run Monte Carlo experiments for greedy Cotlar--Stein selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_C_VALUES,
    DEFAULT_K_VALUES,
    DEFAULT_N_VALUES,
    AlphaFunction,
    get_alpha_function,
)
from src.greedy import Sampler, run_version1, run_version2
from src.sampling import (
    Field,
    sample_spectral_ball_haar_truncation,
    sample_spectral_ball_rejection,
    sample_spectral_ball_scaled_gaussian,
)

Version = Literal[1, 2]

SAMPLERS: dict[str, Sampler] = {
    "scaled_gaussian": sample_spectral_ball_scaled_gaussian,
    "rejection": sample_spectral_ball_rejection,
    "haar_truncation": sample_spectral_ball_haar_truncation,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a grid of Cotlar--Stein experiments."""

    N_values: list[int] = dataclass_field(
        default_factory=lambda: list(DEFAULT_N_VALUES),
    )
    K_values: list[int] = dataclass_field(
        default_factory=lambda: list(DEFAULT_K_VALUES),
    )
    c_values: list[float] = dataclass_field(
        default_factory=lambda: list(DEFAULT_C_VALUES),
    )
    trials: int = 100
    versions: list[Version] = dataclass_field(default_factory=lambda: [1])
    alpha_names: list[str] = dataclass_field(default_factory=lambda: ["linear"])
    sampler_name: str = "scaled_gaussian"
    field_values: list[Field] = dataclass_field(default_factory=lambda: ["complex"])
    max_draws: int = 100_000
    random_seed: int = 123


def validate_config(config: ExperimentConfig) -> None:
    """Validate an experiment configuration."""
    if config.trials <= 0:
        msg = f"trials must be positive, got {config.trials}."
        raise ValueError(msg)
    if config.max_draws <= 0:
        msg = f"max_draws must be positive, got {config.max_draws}."
        raise ValueError(msg)

    for version in config.versions:
        if version not in {1, 2}:
            msg = f"version must be 1 or 2, got {version}."
            raise ValueError(msg)

    if config.sampler_name not in SAMPLERS:
        available = ", ".join(sorted(SAMPLERS))
        msg = f"Unknown sampler {config.sampler_name!r}. Options: {available}."
        raise ValueError(msg)

    for field_name in config.field_values:
        if field_name not in {"real", "complex"}:
            msg = f"field must be 'real' or 'complex', got {field_name!r}."
            raise ValueError(msg)

    if (
        config.sampler_name == "haar_truncation"
        and any(field_name != "complex" for field_name in config.field_values)
    ):
        msg = "haar_truncation sampler currently supports only field='complex'."
        raise ValueError(msg)

    for alpha_name in config.alpha_names:
        get_alpha_function(alpha_name)


def run_single_trial(
    *,
    version: Version,
    N: int,
    K: int,
    c: float,
    alpha_func: AlphaFunction,
    sampler: Sampler,
    field_name: Field,
    rng: np.random.Generator,
    max_draws: int,
) -> dict[str, Any]:
    """Run one trial of Version 1 or Version 2."""
    if version == 1:
        return run_version1(
            N=N,
            K=K,
            alpha_func=alpha_func,
            c=c,
            sampler=sampler,
            field=field_name,
            rng=rng,
        )

    return run_version2(
        N=N,
        K=K,
        alpha_func=alpha_func,
        c=c,
        sampler=sampler,
        field=field_name,
        rng=rng,
        max_draws=max_draws,
    )


def parameter_grid(config: ExperimentConfig) -> Iterable[dict[str, Any]]:
    """Yield parameter combinations from an experiment configuration."""
    for version in config.versions:
        for alpha_name in config.alpha_names:
            for field_name in config.field_values:
                for N in config.N_values:
                    for K in config.K_values:
                        for c in config.c_values:
                            yield {
                                "version": version,
                                "sampler_name": config.sampler_name,
                                "alpha_name": alpha_name,
                                "field": field_name,
                                "N": N,
                                "K": K,
                                "c": c,
                            }


def count_parameter_combinations(config: ExperimentConfig) -> int:
    """Count parameter combinations, excluding Monte Carlo trials."""
    return (
        len(config.versions)
        * len(config.alpha_names)
        * len(config.field_values)
        * len(config.N_values)
        * len(config.K_values)
        * len(config.c_values)
    )


def print_config_summary(config: ExperimentConfig) -> None:
    """Print a short summary of the experiment configuration."""
    combinations = count_parameter_combinations(config)
    total_runs = combinations * config.trials

    print("Experiment configuration:")
    print(f"  N values: {config.N_values}")
    print(f"  K values: {config.K_values}")
    print(f"  c values: {config.c_values}")
    print(f"  versions: {config.versions}")
    print(f"  alpha functions: {config.alpha_names}")
    print(f"  sampler: {config.sampler_name}")
    print(f"  fields: {config.field_values}")
    print(f"  trials per parameter combination: {config.trials}")
    print(f"  parameter combinations: {combinations}")
    print(f"  total Monte Carlo runs: {total_runs}")
    print(f"  max_draws for Version 2: {config.max_draws}")


def confirm_run(config: ExperimentConfig) -> None:
    """Ask for confirmation before running an experiment."""
    print_config_summary(config)
    answer = input("Type 'yes' to start this experiment: ").strip().lower()

    if answer != "yes":
        msg = "Aborted before running experiments."
        raise SystemExit(msg)


def run_experiment_grid(config: ExperimentConfig) -> pd.DataFrame:
    """Run a full Monte Carlo experiment grid and return raw results."""
    validate_config(config)

    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, Any]] = []

    for params in parameter_grid(config):
        alpha_func = get_alpha_function(params["alpha_name"])
        sampler = SAMPLERS[params["sampler_name"]]

        for trial in range(config.trials):
            result = run_single_trial(
                version=params["version"],
                N=params["N"],
                K=params["K"],
                c=params["c"],
                alpha_func=alpha_func,
                sampler=sampler,
                field_name=params["field"],
                rng=rng,
                max_draws=config.max_draws,
            )

            rows.append(
                {
                    **params,
                    "trial": trial,
                    **result,
                }
            )

    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Compute grouped means and standard deviations."""
    if results.empty:
        return pd.DataFrame()

    group_cols = [
        "version",
        "sampler_name",
        "alpha_name",
        "field",
        "N",
        "K",
        "c",
    ]
    value_cols = [
        "accepted_count",
        "draws",
        "sum_norm",
        "A_norm",
        "B_norm",
        "alpha",
        "cotlar_bound",
    ]

    summary = results.groupby(group_cols, dropna=False)[value_cols].agg(
        ["mean", "std"],
    )
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    return summary.reset_index()


def result_paths(
    output_dir: str | Path = "results",
    prefix: str = "cotlar",
) -> tuple[Path, Path]:
    """Return raw and summary CSV paths."""
    output_path = Path(output_dir)
    raw_path = output_path / f"{prefix}_raw_results.csv"
    summary_path = output_path / f"{prefix}_summary_results.csv"
    return raw_path, summary_path


def confirm_overwrite(
    output_dir: str | Path = "results",
    prefix: str = "cotlar",
) -> None:
    """Ask for confirmation before overwriting existing result files."""
    raw_path, summary_path = result_paths(output_dir=output_dir, prefix=prefix)
    existing_paths = [path for path in [raw_path, summary_path] if path.exists()]

    if not existing_paths:
        return

    print("Warning: existing result files were found:")
    for path in existing_paths:
        print(f"  {path}")

    answer = input("Type 'yes' to overwrite these files: ").strip().lower()

    if answer != "yes":
        msg = "Aborted without overwriting existing results."
        raise SystemExit(msg)


def save_results(
    results: pd.DataFrame,
    output_dir: str | Path = "results",
    prefix: str = "cotlar",
) -> tuple[Path, Path]:
    """Save raw and summary results to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_path, summary_path = result_paths(output_dir=output_path, prefix=prefix)
    summary = summarize_results(results)

    results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    return raw_path, summary_path


def main() -> None:
    """Run the experiment configured below."""

    # ============================================================
    # Edit this block for the experiment you want to run.
    # ============================================================
    config = ExperimentConfig(
        N_values=[3],
        K_values=list(range(2, 21)),
        c_values=[1.0],
        trials=20,
        versions=[1],
        alpha_names=["sqrt"],
        sampler_name="rejection",
        field_values=["real"],
        max_draws=10_000,
        random_seed=123,
    )

    output_dir = "results"
    prefix = "cotlar"
    # ============================================================

    confirm_run(config)
    confirm_overwrite(output_dir=output_dir, prefix=prefix)

    results = run_experiment_grid(config)
    raw_path, summary_path = save_results(
        results,
        output_dir=output_dir,
        prefix=prefix,
    )

    print(f"Saved raw results to {raw_path}")
    print(f"Saved summary results to {summary_path}")


if __name__ == "__main__":
    main()
