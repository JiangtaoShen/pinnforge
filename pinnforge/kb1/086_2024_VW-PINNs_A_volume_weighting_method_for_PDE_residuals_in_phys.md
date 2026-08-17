---
slot: 86
title: "VW-PINNs: A volume weighting method for PDE residuals in physics-informed neural networks"
authors: [Jiahao Song, Wenbo Cao, Fei Liao, Weiwei Zhang]
year: 2024
venue: "Acta Mechanica Sinica"
gitrepo: ""
---

## TL;DR
On non-uniform collocation grids, the standard PINN PDE loss `(1/N_r) sum_i |residual|^2` over-weights dense regions and under-weights sparse regions, so PINNs *fail*. VW-PINNs replace this with a **volume-weighted** mean `(1/sum s_i) sum_i s_i |residual|^2` where `s_i` is the domain volume occupied by collocation point `i`, estimated with kernel density estimation to remain meshfree.

## Problem
For external flows (cylinder/airfoil) the collocation cloud must be refined near the body but sparse in the far field. Equal-weight PDE loss then makes the optimiser focus on near-body residuals; far-field residuals never converge and error propagates inward, often causing total training failure.

## Method
Define the volume-weighted residual loss:
$$ \mathcal{L}_r = \frac{1}{\sum_i s_i}\sum_{i=1}^{N_r} s_i \,\big|\partial_t u_\theta(t_i,\mathbf{x}_i) + L[u_\theta] + N[u_\theta]\big|^2 $$
With `s_i` the local "volume". In a mesh setting use cell volumes from the background grid. In meshfree settings, estimate density `p(x_i)` via KDE then set `s_i ~ 1/p(x_i)`:
$$ \hat p(\mathbf{x}_i) = \frac{1}{N_r}\sum_{j=1}^{N_r} K_h(\mathbf{x}_i-\mathbf{x}_j),\quad s_i = \frac{|\Omega|}{N_r \hat p(\mathbf{x}_i)} $$
with Gaussian kernel `K_h(x) = exp(-||x||^2/(2h^2)) / (2 pi h^2)^(d/2)`, bandwidth `h` chosen by Silverman's rule. `s_i` is computed once on the (fixed) collocation set and treated as a constant during training.

Combined with adaptive sampling (RAR / RAD), each newly added point gets a freshly computed `s_i`.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import math

def kde_volumes(X, h=None, vol_total=1.0):
    """X: (N, d) collocation positions; returns s_i (N,)."""
    N, d = X.shape
    if h is None:
        h = (4.0/(d+2))**(1.0/(d+4)) * N**(-1.0/(d+4)) * float(jnp.mean(jnp.std(X, axis=0)))
    diff = X[None, :, :] - X[:, None, :]            # (N, N, d)
    sq   = jnp.sum(diff**2, axis=-1) / (2 * h * h)
    norm = 1.0 / ((2 * math.pi * h * h) ** (d / 2.0))
    p_hat = norm * jnp.mean(jnp.exp(-sq), axis=1)   # (N,)
    s = vol_total / (N * p_hat)
    return s

def vw_loss(params, X, residual_fn, s):
    r = residual_fn(params, X)                      # (N, ...)
    sq = jnp.sum(r**2, axis=-1) if r.ndim > 1 else r**2
    return jnp.sum(s * sq) / jnp.sum(s)

X_col = sample_nonuniform_domain(...)
s = jax.lax.stop_gradient(kde_volumes(X_col, vol_total=domain_volume))

optimizer = optax.adam(1e-3); opt_state = optimizer.init(params)

@jax.jit
def loss_fn(params, X_col, s, X_bc, U_bc, lam_r, lam_bc):
    L_r  = vw_loss(params, X_col, ns_residual, s)
    L_bc = bc_loss(params, X_bc, U_bc)
    return lam_r * L_r + lam_bc * L_bc

@jax.jit
def step(params, opt_state, X_col, s, X_bc, U_bc):
    grads = jax.grad(loss_fn)(params, X_col, s, X_bc, U_bc, 100.0, 1.0)
    upd, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, upd), opt_state

for it in range(N):
    params, opt_state = step(params, opt_state, X_col, s, X_bc, U_bc)
    if it in resample_steps:
        X_col = adaptive_resample(params, X_col)
        s = jax.lax.stop_gradient(kde_volumes(X_col, vol_total=domain_volume))
```

Hyperparameters: 5 x 64 tanh MLP; Adam 3k + L-BFGS 2k (history 100, max inner 20); `lam_r = 100`, `lam_bc = 1`; KDE bandwidth via Silverman; recompute `s_i` after every resampling step; use background mesh volumes when available (cheaper and exact). Volume weighting is *outside* relative-weighting and ill-conditioning controllers (e.g., TSONN) - apply VW first.

## Results
Solves inviscid compressible flow over a circular cylinder (Mach=0.4, far-field radius 40D) and viscous NACA0012 flow at varying inflow conditions where vanilla PINNs fail. On Burgers, VW-PINN combined with adaptive sampling reduces wall-clock for a target relative L2 by ~3x; on Burgers inverse problems, parameter-identification error drops by >1 order of magnitude vs equal-weight PINN.
