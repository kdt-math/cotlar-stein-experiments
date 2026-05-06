# Cotlar-Stein Experiments

This project contains numerical experiments inspired by the improved Cotlar-Stein lemma. The goal is to study greedy selection procedures for random matrices whose pairwise interactions remain controlled in the Cotlar-Stein sense.

The experiments are designed to answer questions such as:

- How many randomly sampled operators can be accepted under a Cotlar-Stein admissibility rule?
- How large is the norm of the accepted operator sum?
- How close is the observed norm of the sum to the Cotlar-Stein upper bound?
- In the stopping-time version, how many random draws are needed to accept `K` operators?

---

## Note on GitHub math formatting

GitHub can be sensitive about how Markdown math is delimited. This README uses fenced `math` blocks for displayed equations:

````markdown
```math
a^2+b^2=c^2
```
````

This is usually more robust on GitHub than using `\[...\]` or deeply indented `$$...$$` blocks, especially inside lists.

---

## Mathematical setup

Let `H` be either a real or complex finite-dimensional Hilbert space:

```math
H = \mathbb{R}^N
\quad \text{or} \quad
H = \mathbb{C}^N.
```

Here `N` is the dimension of the Hilbert space. Operators on `H` are represented by `N x N` matrices.

For a finite ordered family of accepted operators,

```math
\mathcal{S}=(S_1,\ldots,S_r),
```

define two scalar matrices:

```math
A_{\mathcal{S}}
=
\left(\sqrt{\|S_j S_k^*\|_2}\right)_{j,k=1}^r,
\qquad
B_{\mathcal{S}}
=
\left(\sqrt{\|S_j^* S_k\|_2}\right)_{j,k=1}^r.
```

Here:

- `|| . ||_2` is the induced matrix 2-norm, also called the spectral norm.
- `S_k^*` denotes transpose in the real case and conjugate transpose in the complex case.
- The indices `j,k` index the selected operators, not the coordinates of the Hilbert space.

The improved Cotlar-Stein lemma gives:

```math
\left\|\sum_{j=1}^r S_j\right\|_2^2
\le
\|A_{\mathcal{S}}\|_2\,\|B_{\mathcal{S}}\|_2.
```

In these experiments, we usually impose a common admissibility threshold:

```math
\alpha(K,N,c)=\beta(K,N,c).
```

A candidate operator is accepted only if the enlarged family satisfies:

```math
\|A_{\mathcal{S}}\|_2 \le \alpha(K,N,c),
\qquad
\|B_{\mathcal{S}}\|_2 \le \alpha(K,N,c).
```

When `alpha = beta`, the Cotlar-Stein lemma gives the deterministic bound:

```math
\left\|\sum_{S\in\mathcal{S}} S\right\|_2
\le
\alpha(K,N,c).
```

Thus, one of the main numerical questions is how large the observed quantity

```math
\left\|\sum_{S\in\mathcal{S}} S\right\|_2
```

is relative to the allowed Cotlar-Stein scale `alpha(K,N,c)`.

---

## Experiment versions

The project implements two greedy selection procedures.

### Version 1: finite sampling budget

Given `K`, draw exactly `K` candidate operators.

Each candidate is accepted only if appending it to the current selected family keeps both Cotlar-Stein matrices admissible:

```math
\|A_{\mathcal{S}}\|_2 \le \alpha(K,N,c),
\qquad
\|B_{\mathcal{S}}\|_2 \le \alpha(K,N,c).
```

The final accepted family satisfies:

```math
|\mathcal{S}| \le K.
```

Important quantities include:

- `accepted_count`: the number of accepted operators.
- `sum_norm`: the observed norm of the accepted operator sum.
- `A_norm`: the observed value of `||A_S||_2`.
- `B_norm`: the observed value of `||B_S||_2`.

In formula form, `sum_norm` is:

```math
\left\|\sum_{S\in\mathcal{S}} S\right\|_2.
```

### Version 2: sample until `K` operators are accepted

Given `K`, keep drawing candidates until `K` operators are accepted, or until a safety cutoff `max_draws` is reached.

The final accepted family satisfies

```math
|\mathcal{S}|=K
```

if the process terminates successfully.

Important quantities include:

- `draws`: the number of candidate draws used.
- `terminated`: whether the process reached `K` accepted operators before `max_draws`.
- `sum_norm`.
- `A_norm`.
- `B_norm`.

Version 2 can be much slower than Version 1 when the threshold is strict, because the algorithm may reject many candidate operators.

---

## Why these alpha functions?

The threshold `alpha(K,N,c)` controls how much interaction among accepted operators is allowed.

The project currently includes three threshold choices.

### 1. Linear threshold

```math
\alpha(K,N,c)=1+c(K-1).
```

This is the main baseline.

Reasoning:

- For a single operator with norm near 1, the diagonal entries of `A_S` and `B_S` are near 1.
- If many off-diagonal interactions are also of constant size, then the matrix norms can grow roughly linearly in `K`.
- Therefore, a linear threshold is a natural permissive baseline for dense random contractions.

Special cases:

If `c = 1`, then:

```math
\alpha(K,N,1)=K.
```

This is permissive and often accepts many operators. The resulting Cotlar-Stein bound is on the same scale as the trivial triangle inequality:

```math
\left\|\sum_{j=1}^K S_j\right\|_2 \le K.
```

If `0 < c < 1`, the threshold is more selective. It asks the greedy procedure to find families with less accumulated interaction than the fully permissive `K`-scale.

This is the best first threshold to test.

### 2. Square-root threshold

```math
\alpha(K,N,c)=1+c(\sqrt{K}-1).
```

This is a stricter, sublinear threshold.

Reasoning:

- The linear threshold tests whether the family behaves like a generic collection of dense contractions.
- The square-root threshold asks whether the greedy selection rule can find families whose accumulated interactions grow much more slowly than `K`.
- This is more mathematically interesting because the resulting Cotlar-Stein bound is sublinear:

```math
\left\|\sum_{S\in\mathcal{S}} S\right\|_2
\le
1+c(\sqrt{K}-1).
```

The subtraction of 1 is intentional. It gives:

```math
\alpha(1,N,c)=1,
```

which matches the natural scale for a single norm-one operator.

### 3. Constant threshold

```math
\alpha(K,N,c)=c.
```

This is the strongest type of threshold when `c` is fixed independently of `K`.

Reasoning:

- A constant threshold asks for genuinely almost-orthogonal families whose sum remains uniformly bounded as `K` grows.
- This can be hard for dense random operators.
- In Version 2, this may lead to very long runtimes or nontermination before `max_draws`.

This threshold is useful for stress tests and for studying strict almost-orthogonality.

---

## Why use the same alpha and beta?

The improved Cotlar-Stein lemma involves two matrices:

```math
A_{\mathcal{S}}
=
\left(\sqrt{\|S_jS_k^*\|_2}\right),
\qquad
B_{\mathcal{S}}
=
\left(\sqrt{\|S_j^*S_k\|_2}\right).
```

In principle, one could use two different thresholds:

```math
\alpha(K,N,c),
\qquad
\beta(K,N,c).
```

This project currently uses `alpha = beta` for simplicity. This is natural when the random matrix model treats the domain and range symmetrically. It also makes the Cotlar-Stein bound especially simple:

```math
\left\|\sum_{S\in\mathcal{S}} S\right\|_2
\le
\alpha.
```

---

## Samplers

The project samples random matrices satisfying:

```math
\|T\|_2 \le 1.
```

Two samplers are implemented.

### 1. `scaled_gaussian`

This is the fast default sampler.

It samples a Gaussian random matrix and rescales it so that its spectral norm is at most 1.

This sampler is useful for fast experiments, but it is **not uniform** from the spectral-norm unit ball.

Use this sampler for most exploratory runs.

### 2. `rejection`

This is an exact rejection sampler.

It samples uniformly from a containing Frobenius norm ball and rejects candidates whose spectral norm exceeds 1.

This is exact with respect to Lebesgue measure on the spectral-norm unit ball, but it becomes slow as `N` grows.

Use this sampler only for small dimensions, such as `N = 2, 3, 4`.

---

## Project structure

```text
cotlar-stein-experiments/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── sampling.py
│   ├── cotlar.py
│   ├── greedy.py
│   ├── run_experiments.py
│   └── plots.py
├── tests/
│   ├── test_config.py
│   ├── test_sampling.py
│   ├── test_cotlar.py
│   ├── test_greedy.py
│   └── test_run_experiments.py
├── scripts/
│   ├── clean_outputs.sh
│   ├── clean_cache.sh
│   └── clean_all.sh
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Main files

- `src/config.py`: threshold functions and default parameter values.
- `src/sampling.py`: random matrix samplers.
- `src/cotlar.py`: construction of the Cotlar-Stein matrices and related norms.
- `src/greedy.py`: Version 1 and Version 2 greedy selection algorithms.
- `src/run_experiments.py`: Monte Carlo experiment runner.
- `src/plots.py`: plotting utilities.

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run tests and linting

Run all tests:

```bash
pytest
```

Run Ruff:

```bash
ruff check .
```

Auto-fix simple Ruff issues:

```bash
ruff check . --fix
```

---

## Running experiments

Edit the configuration block near the bottom of:

```text
src/run_experiments.py
```

A typical configuration looks like:

```python
config = ExperimentConfig(
    N_values=[5],
    K_values=list(range(2, 21)),
    c_values=[1.0],
    trials=20,
    versions=[1, 2],
    alpha_names=["linear"],
    sampler_name="scaled_gaussian",
    field_values=["real"],
    max_draws=10_000,
    random_seed=123,
)
```

Then run:

```bash
python -m src.run_experiments
```

The script prints a summary and asks for confirmation before starting.

Results are saved to:

```text
results/cotlar_raw_results.csv
results/cotlar_summary_results.csv
```

---

## Meaning of key configuration values

### `N_values`

Dimensions of the Hilbert space.

For example:

```python
N_values=[5]
```

means matrices are `5 x 5`.

### `K_values`

Values of `K` to test.

In Version 1, `K` is the number of candidate draws.

In Version 2, `K` is the target number of accepted operators.

For example:

```python
K_values=list(range(2, 21))
```

tests:

```math
K=2,3,\ldots,20.
```

### `c_values`

Values of the threshold parameter `c`.

For example, with the linear threshold:

```math
\alpha(K,N,c)=1+c(K-1).
```

Thus:

- `c=1.0` gives `alpha = K`.
- `c=0.5` gives `alpha = 1 + 0.5(K-1)`.
- Smaller `c` means a stricter admissibility condition.

### `trials`

Number of Monte Carlo repetitions for each fixed parameter combination.

For example, if `trials=20`, then each fixed choice of

```text
version, sampler, alpha_name, field, N, K, c
```

is repeated 20 times.

The summary CSV reports means and standard deviations over these trials.

### `versions`

Which greedy procedures to run.

```python
versions=[1]
```

runs only Version 1.

```python
versions=[1, 2]
```

runs both versions.

### `alpha_names`

Which threshold functions to use.

Currently available:

```text
linear
sqrt
constant
```

### `sampler_name`

Which sampler to use.

Currently available:

```text
scaled_gaussian
rejection
```

### `field_values`

Which scalar fields to test.

Currently available:

```text
real
complex
```

### `max_draws`

Safety cutoff for Version 2.

If the algorithm does not accept `K` operators within `max_draws` candidate draws, the run stops and records:

```text
terminated = False
```

---

## Output columns

The raw CSV contains columns such as:

```text
version
sampler_name
alpha_name
field
N
K
c
trial
accepted_count
draws
terminated
sum_norm
A_norm
B_norm
alpha
cotlar_bound
```

Important meanings:

- `accepted_count`: number of accepted operators.
- `draws`: total number of candidate draws used.
- `terminated`: whether Version 2 successfully accepted `K` operators before `max_draws`.
- `sum_norm`: observed norm of the accepted operator sum.
- `A_norm`: observed norm of the matrix `A_S`.
- `B_norm`: observed norm of the matrix `B_S`.
- `alpha`: threshold used in the admissibility test.
- `cotlar_bound`: equal to `alpha` when `alpha = beta`.

The summary CSV groups by:

```text
version, sampler_name, alpha_name, field, N, K, c
```

and computes means and standard deviations over trials.

---

## Generating plots

After running experiments, generate plots with:

```bash
python -m src.plots
```

Figures are saved under:

```text
figures/
```

The folder structure is organized by:

```text
field / alpha_name / N / c
```

For example:

```text
figures/
  real/
    linear/
      N_5/
        c_1p0/
          version1_accepted_count_vs_k.png
          version1_sum_norm_vs_k.png
          version2_draws_vs_k.png
          version2_sum_norm_vs_k.png
          sum_norm_over_alpha_vs_k.png
```

The ratio plot

```math
\frac{\left\|\sum S_j\right\|_2}{\alpha}
```

is useful because it shows how close the observed sum norm is to the Cotlar-Stein upper bound.

- Values near 1 indicate the bound is relatively tight.
- Values much smaller than 1 indicate the bound is loose for that experiment.

---

## Cleaning generated files

Remove generated experiment outputs and figures:

```bash
bash scripts/clean_outputs.sh
```

Remove Python, pytest, and Ruff caches:

```bash
bash scripts/clean_cache.sh
```

Remove both outputs and caches:

```bash
bash scripts/clean_all.sh
```

These scripts do not remove `.venv/`.

---

## Adding a new alpha function

To add a new threshold function, edit `src/config.py`.

For example, to add a logarithmic threshold:

```math
\alpha(K,N,c)=1+c\log K,
```

add:

```python
from math import log, sqrt
```

and define:

```python
def alpha_log(K: int, N: int, c: float) -> float:
    \"\"\"Logarithmic threshold alpha(K, N, c) = 1 + c log(K).\"\"\"
    validate_parameters(K, N, c)
    return 1.0 + c * log(K)
```

Then register it:

```python
ALPHA_FUNCTIONS: dict[str, AlphaFunction] = {
    "linear": alpha_linear,
    "sqrt": alpha_sqrt,
    "constant": alpha_constant,
    "log": alpha_log,
}
```

After that, use it in `run_experiments.py`:

```python
alpha_names=["log"]
```

or compare it with existing thresholds:

```python
alpha_names=["linear", "sqrt", "log"]
```

---

## Practical recommendations

For initial experiments, use:

```python
sampler_name="scaled_gaussian"
```

and start with the linear threshold:

```python
alpha_names=["linear"]
c_values=[1.0]
```

This gives a permissive baseline.

Then try stricter values:

```python
c_values=[0.3, 0.5, 0.7, 1.0]
```

or compare threshold regimes:

```python
alpha_names=["linear", "sqrt"]
```

For Version 2, be careful with strict thresholds. If `c` is too small or the threshold is constant, the process may take many draws to accept `K` operators. Increase `max_draws` only after confirming that the experiment is reasonable.

---

## Notes on interpretation

The current experiments are exploratory.

The `scaled_gaussian` sampler is fast and convenient, but it is not uniform from the spectral-norm ball. Therefore, results from this sampler should be interpreted as behavior for a natural random contraction model, not as exact uniform sampling from the spectral-norm ball.

The `rejection` sampler is closer to the stated uniform model, but it is practical only for small `N`.

The most useful comparisons are often:

1. Version 1 accepted count versus `K`.
2. Version 2 draws versus `K`.
3. Sum norm versus `K`.
4. Ratio of observed sum norm to `alpha` versus `K`.
5. Linear threshold versus square-root threshold.
6. Real versus complex fields.

These comparisons help separate three effects:

- how often operators are accepted,
- how large the accepted sum becomes,
- how sharp or loose the Cotlar-Stein bound is for the selected family.
