# Cotlar-Stein Experiments

This repository contains Monte Carlo experiments for studying greedy selection procedures related to the Cotlar-Stein lemma. The code samples random matrices, builds the Cotlar-Stein control matrices, applies greedy admissibility rules, records experiment outcomes, and generates plots organized by experiment parameters.

## Project structure

```text
src/
  config.py           # alpha functions and default experiment parameters
  cotlar.py           # Cotlar-Stein matrix utilities
  greedy.py           # Version 1 and Version 2 greedy algorithms
  sampling.py         # random matrix samplers
  run_experiments.py  # experiment runner and CSV output
  plots.py            # plotting utilities

tests/
  test_config.py
  test_cotlar.py
  test_greedy.py
  test_run_experiments.py
  test_sampling.py
```

The main workflow is:

1. Choose experiment parameters in `src/run_experiments.py`.
2. Run the experiment grid to produce CSV files in `results/`.
3. Run the plotting script to produce figures in `figures/`.

## Alpha functions

The available threshold functions are defined in `src/config.py` and registered in `ALPHA_FUNCTIONS`.

Currently supported choices are:

```text
linear    alpha(K, N, c) = 1 + c(K - 1)
sqrt      alpha(K, N, c) = c sqrt(K)
constant  alpha(K, N, c) = c
log       alpha(K, N, c) = c log(K)
```

The new logarithmic threshold is selected with:

```python
alpha_names=["log"]
```

For example, to run only the logarithmic threshold with `c = 1`, edit the configuration block in `src/run_experiments.py`:

```python
config = ExperimentConfig(
    N_values=[3],
    K_values=list(range(2, 101)),
    c_values=[1.0],
    trials=20,
    versions=[1, 2],
    alpha_names=["log"],
    sampler_name="rejection",
    field_values=["real"],
    max_draws=10_000,
    random_seed=123,
)
```

A useful comparison run is:

```python
alpha_names=["sqrt", "log"]
```

This compares the square-root and logarithmic thresholds while keeping the other parameters fixed.

## Greedy algorithms

The repository currently implements two greedy selection procedures.

### Version 1

Version 1 draws exactly `K` candidates. A candidate is accepted if adding it keeps both Cotlar-Stein matrix norms below the threshold `alpha(K, N, c)`. The final accepted family has size at most `K`.

Important outputs include:

```text
accepted_count
sum_norm
A_norm
B_norm
alpha
cotlar_bound
```

### Version 2

Version 2 draws candidates until `K` matrices have been accepted, or until `max_draws` is reached. This version is useful for measuring how many draws are needed to obtain a full admissible family of size `K`.

Important outputs include:

```text
accepted_count
draws
terminated
sum_norm
A_norm
B_norm
alpha
cotlar_bound
```

## Random matrix samplers

The available samplers are registered in `src/run_experiments.py`.

```text
scaled_gaussian   fast surrogate sampler for the spectral norm unit ball
rejection         exact rejection sampler from a containing Frobenius ball
haar_truncation   exact complex sampler using Haar unitary truncation
```

Notes:

- `rejection` is intended for small dimensions and may become slow as `N` grows.
- `haar_truncation` currently supports only `field="complex"`.
- `scaled_gaussian` is fast but is not uniform on the spectral norm ball.

## Running tests

From the project root, run:

```bash
python -m pytest
```

This checks the alpha functions, matrix utilities, greedy algorithms, samplers, and experiment runner.

## Running experiments

Edit the configuration block in `src/run_experiments.py`, then run:

```bash
python -m src.run_experiments
```

The script prints the experiment configuration and asks for confirmation before starting. It also asks for confirmation before overwriting existing result files.

The output files are:

```text
results/cotlar_raw_results.csv
results/cotlar_summary_results.csv
```

The raw file contains one row per Monte Carlo trial. The summary file groups by experiment parameters and records means and standard deviations.

## Generating plots

After running experiments, generate plots with:

```bash
python -m src.plots
```

Plots are saved under:

```text
figures/<field>/<alpha_name>/N_<N>/c_<c>/
```

For example:

```text
figures/real/log/N_3/c_1p0/
```

## Plot outputs

For each fixed slice of parameters `field`, `alpha_name`, `N`, and `c`, the plotting script creates the following figures when the required data are available.

### Version 1 plots

```text
version1_accepted_count_vs_k.png
version1_sum_norm_vs_k.png
```

### Version 2 plots

```text
version2_draws_vs_k.png
version2_sum_norm_vs_k.png
```

### Combined Version 1 and Version 2 plots

```text
sum_norm_over_alpha_vs_k.png
log_sum_norm_vs_k.png
```

The new plot is:

```text
log_sum_norm_vs_k.png
```

This plot shows the mean of

```text
log(||sum_j S_j||)
```

as a function of `K`, with one curve for each available version.

The logarithm is computed trial by trial before averaging over trials. In other words, the plotted quantity is:

```text
mean(log(sum_norm))
```

not

```text
log(mean(sum_norm))
```

If `sum_norm` is zero for a trial, the log value is stored as `NaN` and ignored by the grouped mean.

## Recommended workflow

A typical workflow is:

```bash
python -m pytest
python -m src.run_experiments
python -m src.plots
```

Then inspect the generated files under `figures/`.

## Notes on interpreting the new log-sum plot

The new `log_sum_norm_vs_k.png` plot is meant to study the growth of the selected operator sum:

```text
||sum_j S_j||
```

Taking the logarithm makes multiplicative growth easier to compare across values of `K`. This is independent of the choice of alpha function. The logarithmic alpha function `alpha(K, N, c) = c log(K)` is a separate experimental threshold, while the log-sum plot is a visualization of the output norm.

