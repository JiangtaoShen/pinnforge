---
slot: 051
title: "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks"
authors: [Chen-Chun Wu, Min Zhu, Qinyang Tan, Y. Kartha, Lu Lu]
year: 2022
venue: Computer Methods in Applied Mechanics and Engineering (arXiv:2207.10289)
gitrepo: ""
---

## TL;DR
A 6000-run benchmark of 10 sampling strategies for PINN residual points: 6 uniform schemes (Grid, Random, LHS, Halton, Hammersley, Sobol), uniform-with-resampling, and three adaptive schemes (RAR-G, and two new ones: **RAD** and **RAR-D**). RAD resamples *all* residual points from a PDF `p(x) ∝ ε(x)^k / E[ε^k] + c`; RAR-D *adds* points sampled from the same PDF. With `k=1, c=1` (RAD) or `k=2, c=0` (RAR-D), the new methods consistently dominate vanilla RAR and uniform sampling on forward and inverse PDEs.

## Problem
Most PINN papers use Grid or Random residual points and never revisit the choice. Previous adaptive methods (RAR-G of Lu 2019; Nabian's `p∝ε`) are either too greedy (over-concentrates points) or too uniform (`k=1, c=0` underperforms on sharp solutions).

## Method
**Loss.** Standard PINN form:
$$ \mathcal{L}_f(\theta;T_f) = \frac{1}{|T_f|}\sum_{x\in T_f} |f(x;\hat u)|^2 $$
with hard BCs where possible; for inverse problems, add `L_i = MSE(û - u_obs)` with weight `w_i`.

**Two new samplers.** Define the residual `ε(x) = |f(x;û)|` and the PDF
$$ p(x) \;\propto\; \frac{\varepsilon^k(x)}{\mathbb{E}[\varepsilon^k(x)]} + c,\qquad k\ge 0,\ c\ge 0 $$

- **RAD (Algorithm 2).** Train for a while; resample *all* of `T_f` from `p(x)`; repeat. `k=0` or `c→∞` recovers uniform random; recommended default `k=1, c=1`.
- **RAR-D (Algorithm 3).** Train; sample `m` new points from `p(x)` and *append* to `T_f`; repeat. `k→∞` recovers RAR-G. Recommended default `k=2, c=0`.

Sampling `p(x)` in low dimensions: draw a dense pool `S_0` (Hammersley/Sobol), compute `p(x)` on it, then draw a sub-sample without replacement using the discrete distribution `p̃(x)=p(x)/Σp`. For higher dims use inverse-transform sampling, MCMC or a GAN.

**Uniform-sampling guidelines (also from the study):**
- Low-discrepancy sequences (Hammersley, Halton, Sobol) ≥ LHS ≥ Random ≥ Grid for the same `N`.
- Random-R (resample every `N_period` Adam steps) almost always beats fixed Random; `N_period` ≈ 100–500.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

def pde_residual(params, x):
    def u_of(z): return model.apply(params, z)
    # ... compute |f(x;u)| for the specific PDE ...
    return f_abs                                            # shape (N,)

def rad_sample(key, params, n_keep, pool, k=1.0, c=1.0):
    eps = pde_residual(params, pool)                        # |residual|
    w   = (eps ** k) / jnp.clip(jnp.mean(eps ** k), a_min=1e-12) + c
    w   = w / w.sum()
    idx = jax.random.choice(key, pool.shape[0], (n_keep,), replace=False, p=w)
    return pool[idx]

def rar_d_sample(key, params, n_add, pool, k=2.0, c=0.0):
    return rad_sample(key, params, n_add, pool, k, c)

T   = sample_hammersley(domain, N=1000)                     # initial set
opt = optax.adam(1e-3)
params    = model.init(jax.random.PRNGKey(0), jnp.zeros((1, d_in)))
opt_state = opt.init(params)

def loss_fn(params, T):
    return jnp.mean(pde_residual(params, T) ** 2)

@jax.jit
def step(params, opt_state, T):
    g = jax.grad(loss_fn)(params, T)
    upd, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state

# Phase 1: train on initial uniform set
for _ in range(20_000):
    params, opt_state = step(params, opt_state, T)

# Phase 2: RAD loop
key = jax.random.PRNGKey(1)
for cycle in range(30):
    key, k_pool, k_samp = jax.random.split(key, 3)
    pool = sample_hammersley(domain, N=100_000)
    T    = rad_sample(k_samp, params, n_keep=2000, pool=pool, k=1.0, c=1.0)
    for _ in range(1000):
        params, opt_state = step(params, opt_state, T)
```

Recommended hyperparameters:
- Uniform baseline: Hammersley / Sobol with 1k–10k points; Random-R period 100–500.
- RAD defaults: `k=1, c=1`; resampling period 1000 iters; pool size 10× target.
- RAR-D defaults: `k=2, c=0`; add ~10% new points per cycle.
- All ablations done with tanh MLPs (e.g. 4×50) trained with Adam (1e-3) + L-BFGS.

## Results
Across four forward problems (Burgers, wave, diffusion-reaction, Allen-Cahn) and two inverse problems (diffusion-reaction, KdV) summarising >6000 runs:
- RAD and RAR-D consistently give 5–50× lower L2 error than RAR-G and uniform sampling at the same `|T_f|`.
- Uniform-with-resampling beats fixed uniform for the same total compute.
- Hammersley/Sobol > Grid > Random among fixed uniform sets.
- `k=1, c=1` and `k=2, c=0` are robust defaults.
