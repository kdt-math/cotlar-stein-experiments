"""Plotting utilities for Cotlar--Stein experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(path: str | Path = "results/cotlar_raw_results.csv") -> pd.DataFrame:
    """Load raw experiment results."""
    return pd.read_csv(path)


def format_c_value(c: float) -> str:
    """Format a c value for folder names.

    Example:
        1.0 -> c_1p0
        0.5 -> c_0p5
    """
    return f"c_{str(c).replace('.', 'p')}"


def format_n_value(N: int) -> str:
    """Format an N value for folder names."""
    return f"N_{N}"


def add_ratio_column(results: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with derived plotting columns added.

    Added columns:
        sum_norm_over_alpha: sum_norm / alpha, set to NaN if alpha is zero.
        log_sum_norm: log(sum_norm), set to NaN if sum_norm is nonpositive.
    """
    copied = results.copy()

    copied["sum_norm_over_alpha"] = np.where(
        copied["alpha"] != 0,
        copied["sum_norm"] / copied["alpha"],
        np.nan,
    )

    copied["log_sum_norm"] = np.nan
    positive_sum_norm = copied["sum_norm"] > 0
    copied.loc[positive_sum_norm, "log_sum_norm"] = np.log(
        copied.loc[positive_sum_norm, "sum_norm"],
    )

    return copied


def mean_by_k(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute the mean of a metric grouped by K."""
    grouped = results.groupby("K", as_index=False)[metric].mean()
    return grouped.sort_values("K").dropna(subset=[metric])


def plot_single_curve_vs_k(
    results: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """Plot one mean curve versus K."""
    if results.empty:
        return

    grouped = mean_by_k(results, metric)
    if grouped.empty:
        return

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


def plot_version_curves_vs_k(
    results: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """Plot mean curves versus K, one curve for each version."""
    if results.empty:
        return

    fig, ax = plt.subplots()
    plotted_any_curve = False

    for version, version_results in results.groupby("version"):
        grouped = mean_by_k(version_results, metric)
        if grouped.empty:
            continue

        ax.plot(
            grouped["K"],
            grouped[metric],
            marker="o",
            label=f"Version {version}",
        )
        plotted_any_curve = True

    if not plotted_any_curve:
        plt.close(fig)
        return

    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_slice(
    results: pd.DataFrame,
    *,
    field: str,
    alpha_name: str,
    N: int,
    c: float,
    output_dir: str | Path = "figures",
) -> None:
    """Create plots for one fixed field, alpha_name, N, and c value."""
    sliced = results[
        (results["field"] == field)
        & (results["alpha_name"] == alpha_name)
        & (results["N"] == N)
        & (results["c"] == c)
    ].copy()

    if sliced.empty:
        return

    sliced = add_ratio_column(sliced)

    slice_dir = (
        Path(output_dir)
        / field
        / alpha_name
        / format_n_value(N)
        / format_c_value(c)
    )

    label = f"{field}, {alpha_name}, N={N}, c={c}"

    version1 = sliced[sliced["version"] == 1]
    version2 = sliced[sliced["version"] == 2]

    plot_single_curve_vs_k(
        version1,
        metric="accepted_count",
        title=f"Version 1: Mean Accepted Count vs K ({label})",
        ylabel="Mean accepted count",
        output_path=slice_dir / "version1_accepted_count_vs_k.png",
    )

    plot_single_curve_vs_k(
        version1,
        metric="sum_norm",
        title=f"Version 1: Mean Sum Norm vs K ({label})",
        ylabel="Mean ||sum S_j||",
        output_path=slice_dir / "version1_sum_norm_vs_k.png",
    )

    plot_single_curve_vs_k(
        version2,
        metric="draws",
        title=f"Version 2: Mean Draws vs K ({label})",
        ylabel="Mean draws",
        output_path=slice_dir / "version2_draws_vs_k.png",
    )

    plot_single_curve_vs_k(
        version2,
        metric="sum_norm",
        title=f"Version 2: Mean Sum Norm vs K ({label})",
        ylabel="Mean ||sum S_j||",
        output_path=slice_dir / "version2_sum_norm_vs_k.png",
    )

    plot_version_curves_vs_k(
        sliced,
        metric="sum_norm_over_alpha",
        title=f"Mean Ratio ||sum S_j|| / alpha vs K ({label})",
        ylabel="Mean ||sum S_j|| / alpha",
        output_path=slice_dir / "sum_norm_over_alpha_vs_k.png",
    )

    plot_version_curves_vs_k(
        sliced,
        metric="log_sum_norm",
        title=f"Mean Log Sum Norm vs K ({label})",
        ylabel="Mean log ||sum S_j||",
        output_path=slice_dir / "log_sum_norm_vs_k.png",
    )


def make_plots_by_slice(
    results_path: str | Path = "results/cotlar_raw_results.csv",
    output_dir: str | Path = "figures",
) -> None:
    """Create plots in folders organized by field, alpha_name, N, and c."""
    results = load_results(results_path)

    required_columns = {"field", "alpha_name", "N", "c"}
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        msg = f"Missing required columns: {sorted(missing_columns)}."
        raise ValueError(msg)

    slice_keys = results[["field", "alpha_name", "N", "c"]].drop_duplicates()

    for row in slice_keys.itertuples(index=False):
        plot_slice(
            results,
            field=row.field,
            alpha_name=row.alpha_name,
            N=row.N,
            c=row.c,
            output_dir=output_dir,
        )


def main() -> None:
    """Generate plots from the default raw results CSV."""
    make_plots_by_slice()
    print("Saved plots to figures/")


if __name__ == "__main__":
    main()
