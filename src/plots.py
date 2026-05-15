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


def custom_bound(K: pd.Series | np.ndarray, N: int) -> pd.Series | np.ndarray:
    """Return the custom comparison bound sqrt(log(2N)) sqrt(K)."""
    return np.sqrt(np.log(2 * N)) * np.sqrt(K)


def ask_include_custom_bound() -> bool:
    """Ask whether to include custom bound curves to sqrt-alpha sum-norm plots."""
    answer = input(
        "Add custom bound curves to sqrt-alpha sum-norm plots? "
        "Type 'yes' or 'no': ",
    ).strip().lower()
    return answer == "yes"


def add_ratio_column(results: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the ratio sum_norm / alpha added.

    If alpha is zero, the ratio is set to NaN.
    """
    copied = results.copy()
    copied["sum_norm_over_alpha"] = np.where(
        copied["alpha"] != 0,
        copied["sum_norm"] / copied["alpha"],
        np.nan,
    )
    return copied


def mean_by_k(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute the mean of a metric grouped by K."""
    grouped = results.groupby("K", as_index=False)[metric].mean()
    return grouped.sort_values("K").dropna(subset=[metric])


def mean_columns_by_k(results: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute means of multiple columns grouped by K."""
    grouped = results.groupby("K", as_index=False)[columns].mean()
    return grouped.sort_values("K").dropna(subset=columns)


def log_mean_by_k(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute log of the mean of a positive metric grouped by K.

    This returns log(mean(metric)) for each K, not mean(log(metric)).
    Nonpositive means are dropped because their logarithms are undefined.
    """
    grouped = mean_by_k(results, metric)
    if grouped.empty:
        return grouped

    grouped = grouped[grouped[metric] > 0].copy()
    if grouped.empty:
        return grouped

    grouped[metric] = np.log(grouped[metric])
    return grouped


def should_add_custom_bound(include_custom_bound: bool, alpha_name: str) -> bool:
    """Return True if the optional custom bound should be shown."""
    return include_custom_bound and alpha_name == "sqrt"


def add_sum_norm_bounds(
    ax: plt.Axes,
    grouped: pd.DataFrame,
    *,
    N: int,
    alpha_name: str,
    include_custom_bound: bool,
) -> None:
    """Add alpha and optional custom bounds to a mean sum-norm plot."""
    ax.plot(
        grouped["K"],
        grouped["alpha"],
        linestyle="--",
        label="alpha bound",
    )

    if should_add_custom_bound(include_custom_bound, alpha_name):
        ax.plot(
            grouped["K"],
            custom_bound(grouped["K"], N),
            linestyle=":",
            label="custom bound",
        )


def add_log_sum_norm_bounds(
    ax: plt.Axes,
    grouped: pd.DataFrame,
    *,
    N: int,
    alpha_name: str,
    include_custom_bound: bool,
) -> None:
    """Add log-alpha and optional log-custom bounds to a log mean plot."""
    positive_alpha = grouped["alpha"] > 0
    if positive_alpha.any():
        ax.plot(
            grouped.loc[positive_alpha, "K"],
            np.log(grouped.loc[positive_alpha, "alpha"]),
            linestyle="--",
            label="log(alpha bound)",
        )

    if should_add_custom_bound(include_custom_bound, alpha_name):
        bound = custom_bound(grouped["K"], N)
        positive_bound = bound > 0
        if positive_bound.any():
            ax.plot(
                grouped.loc[positive_bound, "K"],
                np.log(bound.loc[positive_bound]),
                linestyle=":",
                label="log custom bound sqrt(log(2N)) sqrt(K)",
            )


def plot_single_curve_vs_k(
    results: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    output_path: str | Path,
    show_alpha_bound: bool = False,
    N: int | None = None,
    alpha_name: str | None = None,
    include_custom_bound: bool = False,
) -> None:
    """Plot one mean curve versus K."""
    if results.empty:
        return

    if show_alpha_bound:
        grouped = mean_columns_by_k(results, [metric, "alpha"])
    else:
        grouped = mean_by_k(results, metric)

    if grouped.empty:
        return

    fig, ax = plt.subplots()
    ax.plot(grouped["K"], grouped[metric], marker="o", label="mean")

    if show_alpha_bound:
        if N is None or alpha_name is None:
            msg = "N and alpha_name are required when show_alpha_bound=True."
            raise ValueError(msg)
        add_sum_norm_bounds(
            ax,
            grouped,
            N=N,
            alpha_name=alpha_name,
            include_custom_bound=include_custom_bound,
        )
        ax.legend()

    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_single_log_mean_curve_vs_k(
    results: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    output_path: str | Path,
    show_alpha_bound: bool = False,
    N: int | None = None,
    alpha_name: str | None = None,
    include_custom_bound: bool = False,
) -> None:
    """Plot log(mean(metric)) versus K for one version."""
    if results.empty:
        return

    if show_alpha_bound:
        grouped = mean_columns_by_k(results, [metric, "alpha"])
        grouped = grouped[grouped[metric] > 0].copy()
        if not grouped.empty:
            grouped[metric] = np.log(grouped[metric])
    else:
        grouped = log_mean_by_k(results, metric)

    if grouped.empty:
        return

    fig, ax = plt.subplots()
    ax.plot(grouped["K"], grouped[metric], marker="o", label="log(mean)")

    if show_alpha_bound:
        if N is None or alpha_name is None:
            msg = "N and alpha_name are required when show_alpha_bound=True."
            raise ValueError(msg)
        add_log_sum_norm_bounds(
            ax,
            grouped,
            N=N,
            alpha_name=alpha_name,
            include_custom_bound=include_custom_bound,
        )
        ax.legend()

    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

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
    include_custom_bound: bool = False,
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
    version3 = sliced[sliced["version"] == 3]

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
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=include_custom_bound,
    )

    plot_single_curve_vs_k(
        version1,
        metric="A_norm",
        title=f"Version 1: Mean A Norm vs K ({label})",
        ylabel="Mean ||A||",
        output_path=slice_dir / "version1_A_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_curve_vs_k(
        version1,
        metric="B_norm",
        title=f"Version 1: Mean B Norm vs K ({label})",
        ylabel="Mean ||B||",
        output_path=slice_dir / "version1_B_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_curve_vs_k(
        version1,
        metric="sum_norm_over_alpha",
        title=f"Version 1: Mean Ratio ||sum S_j|| / alpha vs K ({label})",
        ylabel="Mean ||sum S_j|| / alpha",
        output_path=slice_dir / "version1_sum_norm_over_alpha_vs_k.png",
    )

    plot_single_log_mean_curve_vs_k(
        version1,
        metric="sum_norm",
        title=f"Version 1: Log Mean Sum Norm vs K ({label})",
        ylabel="log(mean ||sum S_j||)",
        output_path=slice_dir / "version1_log_mean_sum_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=include_custom_bound,
    )

    plot_single_log_mean_curve_vs_k(
        version1,
        metric="A_norm",
        title=f"Version 1: Log Mean A Norm vs K ({label})",
        ylabel="log(mean ||A||)",
        output_path=slice_dir / "version1_log_mean_A_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_log_mean_curve_vs_k(
        version1,
        metric="B_norm",
        title=f"Version 1: Log Mean B Norm vs K ({label})",
        ylabel="log(mean ||B||)",
        output_path=slice_dir / "version1_log_mean_B_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_log_mean_curve_vs_k(
        version1,
        metric="sum_norm_over_alpha",
        title=f"Version 1: Log Mean Ratio ||sum S_j|| / alpha vs K ({label})",
        ylabel="log(mean ||sum S_j|| / alpha)",
        output_path=slice_dir / "version1_log_mean_sum_norm_over_alpha_vs_k.png",
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
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=include_custom_bound,
    )

    plot_single_curve_vs_k(
        version2,
        metric="A_norm",
        title=f"Version 2: Mean A Norm vs K ({label})",
        ylabel="Mean ||A||",
        output_path=slice_dir / "version2_A_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_curve_vs_k(
        version2,
        metric="B_norm",
        title=f"Version 2: Mean B Norm vs K ({label})",
        ylabel="Mean ||B||",
        output_path=slice_dir / "version2_B_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_curve_vs_k(
        version2,
        metric="sum_norm_over_alpha",
        title=f"Version 2: Mean Ratio ||sum S_j|| / alpha vs K ({label})",
        ylabel="Mean ||sum S_j|| / alpha",
        output_path=slice_dir / "version2_sum_norm_over_alpha_vs_k.png",
    )

    plot_single_log_mean_curve_vs_k(
        version2,
        metric="sum_norm",
        title=f"Version 2: Log Mean Sum Norm vs K ({label})",
        ylabel="log(mean ||sum S_j||)",
        output_path=slice_dir / "version2_log_mean_sum_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=include_custom_bound,
    )

    plot_single_log_mean_curve_vs_k(
        version2,
        metric="A_norm",
        title=f"Version 2: Log Mean A Norm vs K ({label})",
        ylabel="log(mean ||A||)",
        output_path=slice_dir / "version2_log_mean_A_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_log_mean_curve_vs_k(
        version2,
        metric="B_norm",
        title=f"Version 2: Log Mean B Norm vs K ({label})",
        ylabel="log(mean ||B||)",
        output_path=slice_dir / "version2_log_mean_B_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_log_mean_curve_vs_k(
        version2,
        metric="sum_norm_over_alpha",
        title=f"Version 2: Log Mean Ratio ||sum S_j|| / alpha vs K ({label})",
        ylabel="log(mean ||sum S_j|| / alpha)",
        output_path=slice_dir / "version2_log_mean_sum_norm_over_alpha_vs_k.png",
    )

    plot_single_curve_vs_k(
        version3,
        metric="accepted_count",
        title=f"Version 3: Mean Accepted Count vs K ({label})",
        ylabel="Mean accepted count",
        output_path=slice_dir / "version3_accepted_count_vs_k.png",
    )

    plot_single_curve_vs_k(
        version3,
        metric="sum_norm",
        title=f"Version 3: Mean Sum Norm vs K ({label})",
        ylabel="Mean ||sum S_j||",
        output_path=slice_dir / "version3_sum_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=include_custom_bound,
    )

    plot_single_curve_vs_k(
        version3,
        metric="A_norm",
        title=f"Version 3: Mean A Norm vs K ({label})",
        ylabel="Mean ||A||",
        output_path=slice_dir / "version3_A_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_curve_vs_k(
        version3,
        metric="B_norm",
        title=f"Version 3: Mean B Norm vs K ({label})",
        ylabel="Mean ||B||",
        output_path=slice_dir / "version3_B_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_curve_vs_k(
        version3,
        metric="sum_norm_over_alpha",
        title=f"Version 3: Mean Ratio ||sum S_j|| / alpha vs K ({label})",
        ylabel="Mean ||sum S_j|| / alpha",
        output_path=slice_dir / "version3_sum_norm_over_alpha_vs_k.png",
    )

    plot_single_log_mean_curve_vs_k(
        version3,
        metric="sum_norm",
        title=f"Version 3: Log Mean Sum Norm vs K ({label})",
        ylabel="log(mean ||sum S_j||)",
        output_path=slice_dir / "version3_log_mean_sum_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=include_custom_bound,
    )

    plot_single_log_mean_curve_vs_k(
        version3,
        metric="A_norm",
        title=f"Version 3: Log Mean A Norm vs K ({label})",
        ylabel="log(mean ||A||)",
        output_path=slice_dir / "version3_log_mean_A_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_log_mean_curve_vs_k(
        version3,
        metric="B_norm",
        title=f"Version 3: Log Mean B Norm vs K ({label})",
        ylabel="log(mean ||B||)",
        output_path=slice_dir / "version3_log_mean_B_norm_vs_k.png",
        show_alpha_bound=True,
        N=N,
        alpha_name=alpha_name,
        include_custom_bound=False,
    )

    plot_single_log_mean_curve_vs_k(
        version3,
        metric="sum_norm_over_alpha",
        title=f"Version 3: Log Mean Ratio ||sum S_j|| / alpha vs K ({label})",
        ylabel="log(mean ||sum S_j|| / alpha)",
        output_path=slice_dir / "version3_log_mean_sum_norm_over_alpha_vs_k.png",
    )


def make_plots_by_slice(
    results_path: str | Path = "results/cotlar_raw_results.csv",
    output_dir: str | Path = "figures",
    include_custom_bound: bool = False,
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
            include_custom_bound=include_custom_bound,
        )


def main() -> None:
    """Generate plots from the default raw results CSV."""
    include_custom_bound = ask_include_custom_bound()
    make_plots_by_slice(include_custom_bound=include_custom_bound)
    print("Saved plots to figures/")


if __name__ == "__main__":
    main()
