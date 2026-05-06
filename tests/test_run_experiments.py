"""Tests for experiment runner."""

import pandas as pd
import pytest

from src.run_experiments import (
    ExperimentConfig,
    run_experiment_grid,
    save_results,
    summarize_results,
)


def test_run_experiment_grid_returns_expected_columns() -> None:
    config = ExperimentConfig(
        N_values=[2],
        K_values=[2],
        c_values=[1.0],
        trials=2,
        versions=[1],
        alpha_names=["linear"],
        sampler_name="scaled_gaussian",
        field_values=["real"],
        random_seed=123,
    )

    results = run_experiment_grid(config)

    expected_columns = {
        "version",
        "sampler_name",
        "alpha_name",
        "field",
        "N",
        "K",
        "c",
        "trial",
        "accepted_count",
        "draws",
        "terminated",
        "sum_norm",
        "A_norm",
        "B_norm",
        "alpha",
        "cotlar_bound",
    }

    assert not results.empty
    assert expected_columns.issubset(results.columns)
    assert len(results) == 2


def test_version1_accepted_count_is_at_most_k() -> None:
    config = ExperimentConfig(
        N_values=[2],
        K_values=[3],
        c_values=[0.5],
        trials=3,
        versions=[1],
        alpha_names=["linear"],
        sampler_name="scaled_gaussian",
        field_values=["complex"],
        random_seed=123,
    )

    results = run_experiment_grid(config)

    assert (results["accepted_count"] <= results["K"]).all()


def test_version2_accepted_count_equals_k_when_terminated() -> None:
    config = ExperimentConfig(
        N_values=[2],
        K_values=[2],
        c_values=[1.0],
        trials=3,
        versions=[2],
        alpha_names=["linear"],
        sampler_name="scaled_gaussian",
        field_values=["real"],
        max_draws=1_000,
        random_seed=123,
    )

    results = run_experiment_grid(config)
    terminated = results[results["terminated"]]

    assert not terminated.empty
    assert (terminated["accepted_count"] == terminated["K"]).all()


def test_summarize_results_returns_grouped_statistics() -> None:
    config = ExperimentConfig(
        N_values=[2],
        K_values=[2],
        c_values=[1.0],
        trials=2,
        versions=[1],
        alpha_names=["linear"],
        sampler_name="scaled_gaussian",
        field_values=["real"],
        random_seed=123,
    )

    results = run_experiment_grid(config)
    summary = summarize_results(results)

    assert not summary.empty
    assert "accepted_count_mean" in summary.columns
    assert "sum_norm_mean" in summary.columns


def test_save_results_creates_csv_files(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {
                "version": 1,
                "sampler_name": "scaled_gaussian",
                "alpha_name": "linear",
                "field": "real",
                "N": 2,
                "K": 2,
                "c": 1.0,
                "trial": 0,
                "accepted_count": 2,
                "draws": 2,
                "terminated": True,
                "sum_norm": 1.0,
                "A_norm": 1.0,
                "B_norm": 1.0,
                "alpha": 2.0,
                "cotlar_bound": 2.0,
            }
        ]
    )

    raw_path, summary_path = save_results(results, output_dir=tmp_path, prefix="test")

    assert raw_path.exists()
    assert summary_path.exists()


def test_invalid_sampler_name_raises_value_error() -> None:
    config = ExperimentConfig(
        sampler_name="not-a-sampler",
    )

    with pytest.raises(ValueError, match="Unknown sampler"):
        run_experiment_grid(config)
