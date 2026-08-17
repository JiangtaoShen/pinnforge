---
slot: 118
title: "NPINN+: an enhanced physics-informed neural network for solving wave equations with nonlocal boundary conditions"
authors: [Qiancheng Tan, Shuyun Yang, Yonghui Qin]
year: 2026
venue: Scientific Reports
gitrepo: ""
doi: 10.1038/s41598-026-50374-9
---

## TL;DR
NPINN+ solves wave equations with global-integral (nonlocal) boundary constraints by (i) reformulating the problem into a wave equation with Neumann BCs plus an integral source term via a closed-form change of variable, (ii) Sobol-sequence + residual-driven adaptive sampling, and (iii) SoftAdapt dynamic loss weighting that up-weights slow-decaying losses.

## Problem
Wave PDEs with a nonlocal mass constraint `∫_Ω V dx = E(t)` produce non-symmetric operators that classical PINNs handle poorly: the integral constraint couples all collocation points globally, gradients of the five-term loss become unbalanced, and uniformly sampled residual points miss boundary-localised error. The authors target 1D/2D/3D wave equations with nonlocal conditions on rectangular and star-shaped (polar-transformed) domains.

## Method

### A. Equivalent local reformulation
Introduce `U(x,t) = V(x,t) + (2l)^{-d} Σ_s x_s² ∫_Ω Q dΩ`. Then (1.1)+(1.2) becomes a wave equation with Neumann-type periodic-flux conditions and modified source `P` plus initial data `ω, γ` — fully local.

$$\partial_t^2 U - \Delta U = P(x,t),\quad U|_{x_s=0}=0,\ [\nabla U\cdot n]_{x_s=0}=[\nabla U\cdot n]_{x_s=l}$$

### B. Composite loss with derivative matching + nonlocal residual
Five terms: PDE residual `L_w`, Dirichlet `L_b`, Neumann-periodic `L_n` (rolled into `L_b`), IC `L_ic`, derivative-supervision `L_p` (matches `∂_t, ∂_{x_s}, ∂_t², ∂_{x_s}²` against the analytic transformed field where available), and nonlocal-residual `L_NL` enforcing the rewritten constraint.

### C. SoftAdapt weights (key adaptive piece)
With `Δ_κ(e)=L_κ(e-1)-L_κ(e)`:
$$\omega_\kappa(e)=\frac{\exp(-\alpha\,\Delta_\kappa(e))}{\sum_{j\in K}\exp(-\alpha\,\Delta_j(e))},\qquad L_{\text{tot}}=\sum_\kappa \omega_\kappa L_\kappa$$
Recommended `α = 3`.

### D. Residual-driven dynamic sampling
Generate Sobol global points; each iteration evaluate residual, take top τ=0.1% high-residual points, sample n local Uniform neighbors of half-width δ, union with fresh Sobol global points, truncate to budget (~30k for 3D, ~12k for 1D).

```python
import jax, jax.numpy as jnp
import optax
import scipy.stats.qmc as qmc

def softadapt(losses_prev, losses_now, alpha=3.0):
    deltas = losses_prev - losses_now
    return jax.nn.softmax(-alpha * deltas)

def adaptive_resample(params, apply_fn, X_pool, pde_residual_fn,
                      top_frac=0.001, n_local=10, delta=0.05,
                      n_global=10000, d=2, key=jax.random.PRNGKey(0)):
    r = jnp.abs(pde_residual_fn(params, apply_fn, X_pool)).squeeze()
    k = max(1, int(top_frac * X_pool.shape[0]))
    idx = jnp.argsort(-r)[:k]
    seeds = X_pool[idx]
    key, sk = jax.random.split(key)
    noise = (jax.random.uniform(sk, (k, n_local, d+1)) * 2 - 1) * delta
    local = jnp.clip((seeds[:, None, :] + noise).reshape(-1, d+1), 0.0, 1.0)
    sob = jnp.asarray(qmc.Sobol(d=d+1, seed=1234).random(n_global),
                      dtype=jnp.float32)
    return jnp.concatenate([local, sob], axis=0)

# Training loop
opt = optax.chain(optax.adam(1e-3))
opt_state = opt.init(params)
prev = None
for it in range(20000):
    losses = compute_all_losses(params, apply_fn, X)    # (L_w, L_b, L_ic, L_p, L_NL)
    if prev is None or it < 500:
        w = jnp.ones(5) / 5
    else:
        w = softadapt(prev, jax.lax.stop_gradient(losses), alpha=3.0)
    def total(p):
        Ls = compute_all_losses(p, apply_fn, X)
        return jnp.sum(w * Ls)
    g = jax.grad(total)(params)
    u, opt_state = opt.update(g, opt_state, params)
    params = optax.apply_updates(params, u)
    prev = jax.lax.stop_gradient(losses)
    if it % 1000 == 999:
        X = adaptive_resample(params, apply_fn, X, pde_residual_fn)
# Stage 2: jaxopt.LBFGS with strong-Wolfe line search, max 50000 iters
```

Hyperparameters: 7 hidden layers x 100 neurons, tanh, Xavier (`nn.initializers.glorot_normal`), `optax.adam(1e-3)` with scale-by schedule (γ=0.9 every 1000), then L-BFGS strong-Wolfe (`jaxopt.LBFGS`), seed=1234. Collocation: 12k for `L_w,L_p` and 3k for `L_b,L_ic` in 1D; scale up to 30k/9k in 3D.

## Results
On 1D/2D/3D wave equations with passive/active source terms and on disk/flower-shaped polar-transformed domains, NPINN+ attains MAE in 10^-4-10^-3 range — roughly 5-10x better than vanilla PINN, with NPINN (no SoftAdapt + no adaptive sampling) in between. Adaptive sampling stabilizes after ~3 outer rounds with ~4k points; SoftAdapt α=3 gives the fastest loss decay.
