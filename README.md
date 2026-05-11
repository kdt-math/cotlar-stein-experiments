# Cotlar-Stein Experiments

This project contains numerical experiments inspired by the improved Cotlar-Stein lemma. The goal is to study greedy selection procedures for random matrices whose pairwise interactions remain controlled in the Cotlar-Stein sense.

The experiments are designed to answer questions such as:

- How many randomly sampled operators can be accepted under a Cotlar-Stein admissibility rule?
- How large is the norm of the accepted operator sum?
- How does the logarithm of the accepted sum norm grow with `K`?
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

The project currently includes four threshold choices.

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

### 4. Logarithmic threshold

```math
\alpha(K,N,c)=c\log K.
```

This is a stricter sublinear threshold than the square-root threshold for large `K`.

Reasoning:

- The logarithmic threshold tests whether the greedy selection rule can find families whose Cotlar-Stein interactions grow very slowly with `K`.
- It is useful for experiments focused on highly restrictive almost-orthogonality.
- Since `log(1)=0`, this threshold is most natural when the experiment starts at `K >= 2`.

Special case:

If `c = 1`, then:

```math
\alpha(K,N,1)=\log K.
```

This is the main logarithmic threshold currently used for exploratory runs with `c_values=[1.0]`.

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

There are three implemented samplers. They have different mathematical meanings.

### 1. `scaled_gaussian`: fast random contractions

This is the fast default sampler.

The sampler does the following:

1. Draw a random Gaussian matrix `G`.
2. Compute its induced matrix 2-norm, also called the spectral norm.
3. Rescale the matrix so that its spectral norm is at most 1.

In formula form, the output is approximately of the form:

```math
T = \rho \frac{G}{\|G\|_2},
\qquad
0 \le \rho \le 1.
```

Therefore,

```math
\|T\|_2
=
\rho
\left\|\frac{G}{\|G\|_2}\right\|_2
=
\rho
\le
1.
```

So every sampled matrix satisfies the required contraction condition:

```math
\|T\|_2 \le 1.
```

However, this sampler is **not uniform** from the spectral-norm unit ball.

The reason is that Gaussian matrices have their own preferred distribution of directions. After normalizing by `||G||_2`, the matrix lands on the spectral-norm boundary, but the directions are not uniformly distributed over that boundary. The radial factor does not fix this directional bias.

Thus, `scaled_gaussian` should be interpreted as a fast random contraction model, not exact uniform sampling from the spectral-norm ball.

This sampler is useful for exploratory experiments because it is fast and always returns a valid contraction.

### 2. `rejection`: exact uniform sampling from the spectral-norm ball

This sampler is designed to sample uniformly from the spectral-norm unit ball:

```math
\mathbb{B}_2
=
\{T : \|T\|_2 \le 1\}.
```

Direct uniform sampling from this set is difficult because the spectral-norm ball has a complicated shape.

Instead, the code samples from a larger and simpler ball: the Frobenius-norm ball of radius `sqrt(N)`,

```math
\mathbb{B}_F(\sqrt{N})
=
\{T : \|T\|_F \le \sqrt{N}\}.
```

This Frobenius ball contains the spectral-norm unit ball. Indeed,

```math
\|T\|_F \le \sqrt{N}\,\|T\|_2.
```

Therefore, if `||T||_2 <= 1`, then:

```math
\|T\|_F \le \sqrt{N}.
```

Hence:

```math
\mathbb{B}_2
\subseteq
\mathbb{B}_F(\sqrt{N}).
```

The rejection sampler then does this:

1. Sample a matrix uniformly from the Frobenius ball `B_F(sqrt(N))`.
2. Check whether `||T||_2 <= 1`.
3. If yes, accept the matrix.
4. If no, reject it and try again.

This gives exact uniform sampling from the spectral-norm ball by the usual rejection-sampling principle: if one samples uniformly from a larger set and accepts exactly the points that lie in a smaller subset, then the accepted points are uniformly distributed on that smaller subset.

The downside is speed. The Frobenius ball is much larger than the spectral-norm ball, especially as `N` grows. Most proposals may have spectral norm greater than 1 and therefore get rejected.

For this reason, `rejection` is mathematically closer to exact uniform sampling, but it is practical only for small dimensions, such as `N = 2, 3, 4`.

### 3. `haar_truncation`: exact complex uniform sampling from the spectral-norm ball

This sampler is an exact uniform sampler for the **complex** spectral-norm unit ball:

```math
\mathbb{B}_2(\mathbb{C}^{N\times N})
=
\{T \in \mathbb{C}^{N\times N} : \|T\|_2 \le 1\}.
```

It uses a Haar-unitary truncation construction:

1. Generate a Haar-distributed unitary matrix `U` of size `2N x 2N`.
2. Return the upper-left `N x N` block of `U`.

In formula form, if

```math
U \in \mathbb{C}^{2N\times 2N}
```

is Haar-distributed and partitioned as

```math
U =
\begin{bmatrix}
T & * \\
* & *
\end{bmatrix},
```

then the returned block `T` satisfies

```math
\|T\|_2 \le 1
```

and is uniformly distributed in the complex spectral-norm unit ball.

This sampler is usually much faster than rejection sampling. It is the recommended exact sampler when `field_values=["complex"]`.

Current limitation:

- `haar_truncation` supports only `field="complex"`.
- It should not be used with `field_values=["real"]`.

To use it in experiments, set:

```python
sampler_name="haar_truncation"
field_values=["complex"]
```

### Summary of sampler choices

| Sampler | Meaning | Uniform from spectral-norm ball? | Speed |
|---|---|---:|---:|
| `scaled_gaussian` | Fast random contraction model | No | Fast |
| `rejection` | Exact uniform spectral-ball sampler | Yes | Slow |
| `haar_truncation` | Exact complex uniform spectral-ball sampler | Yes, complex only | Fast |

Recommended workflow:

1. Use `scaled_gaussian` for fast exploratory runs.
2. Use `haar_truncation` for exact uniform sampling in the complex case.
3. Use `rejection` only for small `N`, especially when testing or comparing the real case.
4. Compare samplers at small dimensions to see whether the qualitative behavior changes.

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
log
```

### `sampler_name`

Which sampler to use.

Currently available:

```text
scaled_gaussian
rejection
haar_truncation
```

The `haar_truncation` sampler currently supports only the complex field, so use it with:

```python
field_values=["complex"]
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
          log_sum_norm_vs_k.png
```

The ratio plot

```math
\frac{\left\|\sum S_j\right\|_2}{\alpha}
```

is useful because it shows how close the observed sum norm is to the Cotlar-Stein upper bound.

- Values near 1 indicate the bound is relatively tight.
- Values much smaller than 1 indicate the bound is loose for that experiment.

The log-sum-norm plot

```math
\log\left\|\sum S_j\right\|_2
```

is useful for studying the growth rate of the accepted operator sum. The code computes `log_sum_norm` trial-by-trial and then plots the mean value over trials for each `K`. If `sum_norm` is zero, the logarithm is recorded as `NaN` and omitted from the plotted mean.

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

## Adding another alpha function

To add another threshold function, edit `src/config.py`.

The logarithmic threshold is already included:

```math
\alpha(K,N,c)=c\log K.
```

It is available through:

```python
alpha_names=["log"]
```

For a future custom threshold, define a new function with the same signature:

```python
def alpha_custom(K: int, N: int, c: float) -> float:
    """Custom threshold alpha(K, N, c)."""
    validate_parameters(K, N, c)
    return ...
```

Then register it:

```python
ALPHA_FUNCTIONS: dict[str, AlphaFunction] = {
    "linear": alpha_linear,
    "sqrt": alpha_sqrt,
    "constant": alpha_constant,
    "log": alpha_log,
    "custom": alpha_custom,
}
```

After that, use it in `run_experiments.py`:

```python
alpha_names=["custom"]
```

or compare it with existing thresholds:

```python
alpha_names=["linear", "sqrt", "log", "custom"]
```

## Practical recommendations

For initial exploratory experiments, use:

```python
sampler_name="scaled_gaussian"
```

For exact uniform complex spectral-norm-ball experiments, use:

```python
sampler_name="haar_truncation"
field_values=["complex"]
```

Then start with the linear threshold:

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
alpha_names=["linear", "sqrt", "log"]
```

For Version 2, be careful with strict thresholds. If `c` is too small or the threshold is constant, the process may take many draws to accept `K` operators. Increase `max_draws` only after confirming that the experiment is reasonable.

---

## Notes on interpretation

The current experiments are exploratory.

The `scaled_gaussian` sampler is fast and convenient, but it is not uniform from the spectral-norm ball. Therefore, results from this sampler should be interpreted as behavior for a natural random contraction model, not as exact uniform sampling from the spectral-norm ball.

The `haar_truncation` sampler gives exact uniform samples from the complex spectral-norm ball and is the preferred exact sampler for complex experiments.

The `rejection` sampler is also exact, but it is practical only for small `N`. It remains useful for small-dimensional checks and for real-field experiments.

The most useful comparisons are often:

1. Version 1 accepted count versus `K`.
2. Version 2 draws versus `K`.
3. Sum norm versus `K`.
4. Log sum norm versus `K`.
5. Ratio of observed sum norm to `alpha` versus `K`.
6. Linear threshold versus square-root and logarithmic thresholds.
7. Real versus complex fields.

These comparisons help separate three effects:

- how often operators are accepted,
- how large the accepted sum becomes,
- how sharp or loose the Cotlar-Stein bound is for the selected family.
