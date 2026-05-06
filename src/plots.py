"""Plotting utilities for Cotlar--Stein experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path: str | Path = "results/cotlar_raw_results.csv") -> pd.DataFrame:
    """Load raw experiment results."""
    return pd.read_csv(path)


def filter_results(
    results: pd.DataFrame,
    *,
    version: int | None = None,
    sampler_name: str | None = None,
    alpha_name: str | None = None,
    field: str | None = None,
    N: int | None = None,
    c: float | None = None,
) -> pd.DataFrame:
    """Filter results by common experiment parameters."""
    filtered = results.copy()

    if version is not None:
        filtered = filtered[filtered["version"] == version]
    if sampler_name is not None:
        filtered = filtered[filtered["sampler_name"] == sampler_name]
    if alpha_name is not None:
        filtered = filtered[filtered["alpha_name"] == alpha_name]
    if field is not None:
        filtered = filtered[filtered["field"] == field]
    if N is not None:
        filtered = filtered[filtered["N"] == N]
    if c is not None:
        filtered = filtered[filtered["c"] == c]

    return filtered


def plot_mean_metric_vs_k(
    results: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """Plot the mean of a metric versus K."""
    grouped = results.groupby("K", as_index=False)[metric].mean()

    fig, ax = plt.subplots()
    ax.plot(grouped["K"], grouped[metric], marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def add_ratio_column(results: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the ratio sum_norm / alpha added."""
    copied = results.copy()
    copied["sum_norm_over_alpha"] = copied["sum_norm"] / copied["alpha"]
    return copied


def make_default_plots(
    results_path: str | Path = "results/cotlar_raw_results.csv",
    output_dir: str | Path = "figures",
) -> None:
    """Create a small collection of default plots."""
    results = load_results(results_path)
    output_dir = Path(output_dir)

    version1 = filter_results(results, version=1)
    version2 = filter_results(results, version=2)

    if not version1.empty:
        plot_mean_metric_vs_k(
            version1,
            metric="accepted_count",
            title="Version 1: Mean Accepted Count vs K",
            ylabel="Mean accepted count",
            output_path=output_dir / "version1_accepted_count_vs_k.png",
        )

        plot_mean_metric_vs_k(
            version1,
            metric="sum_norm",
            title="Version 1: Mean Sum Norm vs K",
            ylabel="Mean ||sum S_j||",
            output_path=output_dir / "version1_sum_norm_vs_k.png",
        )

    if not version2.empty:
        plot_mean_metric_vs_k(
            version2,
            metric="draws",
            title="Version 2: Mean Draws vs K",
            ylabel="Mean draws",
            output_path=output_dir / "version2_draws_vs_k.png",
        )

    ratio_results = add_ratio_column(results)
    plot_mean_metric_vs_k(
        ratio_results,
        metric="sum_norm_over_alpha",
        title="Mean Ratio ||sum S_j|| / alpha vs K",
        ylabel="Mean ratio",
        output_path=output_dir / "sum_norm_over_alpha_vs_k.png",
    )


def main() -> None:
    """Generate default plots from the default raw results CSV."""
    make_default_plots()
    print("Saved plots to figures/")


if __name__ == "__main__":
    main()
