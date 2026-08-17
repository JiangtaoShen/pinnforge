---
slot: 60
title: "Mitigating Propagation Failures in PINNs using Retain-Resample-Release (R3) Sampling"
authors: [Arka Daw, Jie Bu, Sifan Wang, Paris Perdikaris, Anuj Karpatne]
year: 2022
venue: arXiv:2207.02338 (ICML 2023 workshop / arXiv)
gitrepo: "https://github.com/arkadaw9/r3_sampling_icml2023"
---

## TL;DR
PINNs fail when the **correct IC/BC solution cannot propagate** into the interior — local trivial-solution attractors create high-residual "barriers" with skewed/heavy-tailed residual distributions. R3 sampling maintains a fixed-size collocation pool and, every iteration, **retains** points with above-mean residual, **resamples** the rest uniformly, and **releases** retained points once the residual there drops — concentrating effort on barriers without growing the pool.

## Problem
Plain uniform collocation cannot capture narrow, persistent high-residual regions (e.g. convection with `beta=50`, KS chaotic regime), where the residual field becomes heavily skewed and the PINN gets stuck. RAR-style adaptive refinement needs a huge dense candidate pool (100k-1M points) and grows the training set every K iters — slow and biased. Higher-order `L_p` losses (`p=infty`) cause oscillation between residual peaks.

## Method
Keep a constant population `P_i` of `N_r` collocation points. Each iteration:

1. Evaluate residual function `F(x) = |R_theta(x)|` on every `x in P_i`.
2. Compute threshold `tau_i = (1/N_r) sum_x F(x)` (the mean).
3. **Retained set** `P_i^r = {x in P_i : F(x) > tau_i}`.
4. **Resampled set** `P_i^s` of size `N_r - |P_i^r|` drawn from `U(Omega)`.
5. `P_{i+1} = P_i^r ∪ P_i^s`. Run one PINN gradient step on `P_{i+1}`.

Properties (proved in the paper):
- Retain: as iterations advance, the retained pool concentrates where residuals are highest, approaching the L_infty regime in the limit.
- Resample: `|P_i^s| > 0` always — non-zero uniform support prevents collapse.
- Release: once the network resolves a barrier, those points drop below `tau_i` and are released back into the uniform pool.

**Causal R3** (for time-dependent PDEs): instead of one global threshold, sort collocation points by `t` and apply R3 to a *growing* time window — accept retained points only inside the current causal frontier `[0, t_front]`; expand the frontier as residuals there drop. Compatible with the causal-weight loss of Wang et al. 2022 (slot 060).

PINN loss is standard:
$$
\mathcal L(\theta) = \lambda_r\,\frac{1}{N_r}\sum_{x \in P_i}|R_\theta(x)|^2 + \lambda_{ic}\mathcal L_{ic} + \lambda_{bc}\mathcal L_{bc}
$$

```python
import jax, jax.numpy as jnp
import optax

def sample_uniform(key, N, dom_lo, dom_hi):
    d = dom_lo.size
    return dom_lo + (dom_hi - dom_lo) * jax.random.uniform(key, (N, d))

def residual_abs(params, apply_fn, X):                # |R_theta(x)| per row
    r = jax.vmap(lambda x: pde_residual(params, apply_fn, x))(X)
    return jnp.abs(r).reshape(-1)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def pinn_step(params, opt_state, P, X_ic, U_ic, X_bc, U_bc):
    def loss(p):
        r = jax.vmap(lambda x: pde_residual(p, apply_fn, x))(P)
        L_pde = jnp.mean(r**2)
        L_ic  = ic_loss(p, X_ic, U_ic)
        L_bc  = bc_loss(p, X_bc, U_bc)
        return lam_r * L_pde + lam_ic * L_ic + lam_bc * L_bc
    grads = jax.grad(loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

key = jax.random.PRNGKey(0)
P   = sample_uniform(key, N_r, dom_lo, dom_hi)

for it in range(max_iter):
    params, opt_state = pinn_step(params, opt_state, P, X_ic, U_ic, X_bc, U_bc)
    # R3 update of the pool
    F   = residual_abs(params, apply_fn, P)
    tau = jnp.mean(F)
    keep_mask = F > tau
    P_r = P[keep_mask]
    n_s = N_r - P_r.shape[0]
    key, sub = jax.random.split(key)
    P_s = sample_uniform(sub, int(n_s), dom_lo, dom_hi)
    P   = jnp.concatenate([P_r, P_s], axis=0)
```

Hyper-params: `N_r ∈ [1e3, 1e4]` (much smaller than RAR's `P_dense`), Adam(1e-3), 4-6 hidden layers x 128 tanh.

## Results
On convection (`beta=30..50`), reaction-diffusion, Allen-Cahn, and 1-D KS, R3 matches or beats RAR, RAD, RAR-D, `L_infty` PINN, and SA-PINN at 10-100x fewer residual evaluations per iteration. Causal R3 plus the causal weighted loss extends success to KS in the chaotic regime.
