"""Tests for experiment runner."""

import pandas as pd
import pytest

from src.run_experiments import (
    SAMPLERS,
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


def test_version3_accepted_count_always_equals_k() -> None:
    config = ExperimentConfig(
        N_values=[2],
        K_values=[3],
        c_values=[1.0],
        trials=3,
        versions=[3],
        alpha_names=["linear"],
        sampler_name="scaled_gaussian",
        field_values=["real"],
        random_seed=123,
    )

    results = run_experiment_grid(config)

    assert not results.empty
    assert (results["version"] == 3).all()
    assert (results["accepted_count"] == results["K"]).all()
    assert results["terminated"].all()


def test_run_experiment_grid_accepts_all_three_versions() -> None:
    config = ExperimentConfig(
        N_values=[2],
        K_values=[2],
        c_values=[1.0],
        trials=2,
        versions=[1, 2, 3],
        alpha_names=["linear"],
        sampler_name="scaled_gaussian",
        field_values=["real"],
        max_draws=1_000,
        random_seed=123,
    )

    results = run_experiment_grid(config)

    assert not results.empty
    assert set(results["version"]) == {1, 2, 3}
    assert len(results) == 6


def test_invalid_version_raises_value_error() -> None:
    config = ExperimentConfig(
        versions=[4],  # type: ignore[list-item]
    )

    with pytest.raises(ValueError, match="version must be 1, 2, or 3"):
        run_experiment_grid(config)


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
                "cotlar_bound": 1.0,
            }
        ]
    )

    raw_path, summary_path = save_results(results, output_dir=tmp_path, prefix="test")

    assert raw_path.exists()
    assert summary_path.exists()


def test_haar_truncation_sampler_is_registered() -> None:
    assert "haar_truncation" in SAMPLERS


def test_run_experiment_grid_accepts_haar_truncation_sampler() -> None:
    config = ExperimentConfig(
        N_values=[2],
        K_values=[2],
        c_values=[1.0],
        trials=2,
        versions=[1],
        alpha_names=["linear"],
        sampler_name="haar_truncation",
        field_values=["complex"],
        random_seed=123,
    )

    results = run_experiment_grid(config)

    assert not results.empty
    assert len(results) == 2
    assert (results["sampler_name"] == "haar_truncation").all()
    assert (results["field"] == "complex").all()
    assert (results["accepted_count"] <= results["K"]).all()


def test_haar_truncation_sampler_rejects_real_field_in_config() -> None:
    config = ExperimentConfig(
        sampler_name="haar_truncation",
        field_values=["real"],
    )

    with pytest.raises(ValueError, match="field='complex'"):
        run_experiment_grid(config)


def test_invalid_sampler_name_raises_value_error() -> None:
    config = ExperimentConfig(
        sampler_name="not-a-sampler",
    )

    with pytest.raises(ValueError, match="Unknown sampler"):
        run_experiment_grid(config)
