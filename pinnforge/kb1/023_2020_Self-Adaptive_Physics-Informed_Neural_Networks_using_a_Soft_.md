---
slot: 023
title: "Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism"
authors: [Levi D. McClenny, Ulisses Braga-Neto]
year: 2020
venue: "AAAI Spring Symp. MLPS / J. Comp. Phys. (arXiv:2009.04544)"
gitrepo: "https://github.com/levimcclenny/SA-PINNs"
---

## TL;DR
Give every collocation, BC and IC point its OWN trainable weight lambda_i. Train network weights w by gradient descent and the weights lambda by gradient ASCENT on the same loss — yielding a soft per-point attention mask that automatically inflates around stubborn regions (sharp fronts, shocks).

## Problem
Per-loss-term scalar weighting (Wang 2021, NTK weighting, GradNorm) cannot represent intra-term imbalance: when a stiff PDE has a hard region in the interior or a tough IC point, the scalar lambda_r cannot focus on it. Allen-Cahn, advection, and wave PDEs fail at vanilla PINN.

## Method
Replace L_r = (1/N) sum |r_i|^2 by
$$
L_r(w,\lambda_r) = \tfrac{1}{2}\sum_{i=1}^{N_r} m(\lambda_r^i)\,|\mathcal{N}[u_w](x_i^r,t_i^r) - f_i|^2
$$
and analogously L_b, L_0 with per-point weights lambda_b, lambda_0. m: [0,inf) -> [0,inf) is a strictly increasing differentiable mask — polynomial m(lam) = c*lam^q (q in {2,4}) or sigmoid for sharper attention.

Saddle-point training: descent on w, ascent on lambdas
$$
w^{k+1} = w^k - \eta\,\nabla_w L,\quad \lambda^{k+1} = \lambda^k + \rho\,\nabla_\lambda L
$$
Since m'(lam) > 0, lambda only grows where the unmasked loss is non-zero — stubborn points get heavier weight automatically.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

# Per-point weights live OUTSIDE the network params.
lambdas = {
    "r": jnp.ones((N_r,)),
    "b": jnp.ones((N_b,)),
    "0": jnp.ones((N_0,)),
}

def mask(lam, q=2):
    return jnp.clip(lam, a_min=0.0) ** q

def loss_fn(params, lambdas, x_r, t_r, f_r, x_b, t_b, g_b, x_0, h_0):
    r = jax.vmap(lambda x, t: pde_residual(params, x, t))(x_r, t_r) - f_r
    b = jax.vmap(lambda x, t: bc_op(params, x, t))(x_b, t_b) - g_b
    z = jax.vmap(lambda x:    apply_fn(params, x, 0.0))(x_0) - h_0
    L_r = 0.5 * jnp.sum(mask(lambdas["r"]) * r**2)
    L_b = 0.5 * jnp.sum(mask(lambdas["b"]) * b**2)
    L_0 = 0.5 * jnp.sum(mask(lambdas["0"]) * z**2)
    return L_r + L_b + L_0

# Descent on w, ascent on lambdas == descent on -L wrt lambdas.
opt_w   = optax.adam(1e-3)
opt_lam = optax.chain(optax.scale(-1.0), optax.adam(5e-3))   # ascent
state_w   = opt_w.init(params)
state_lam = opt_lam.init(lambdas)

@jax.jit
def step(params, lambdas, state_w, state_lam, batch):
    grads_w   = jax.grad(loss_fn, argnums=0)(params, lambdas, *batch)
    grads_lam = jax.grad(loss_fn, argnums=1)(params, lambdas, *batch)
    upd_w,   state_w   = opt_w.update(grads_w,   state_w,   params)
    upd_lam, state_lam = opt_lam.update(grads_lam, state_lam, lambdas)
    params  = optax.apply_updates(params,  upd_w)
    lambdas = optax.apply_updates(lambdas, upd_lam)
    return params, lambdas, state_w, state_lam
```

For SGD on large collocation sets, train per-point lambdas on a fixed grid and interpolate via a GP to mini-batch points. After Adam, freeze lambdas and finish with L-BFGS (jaxopt).

Recommended: m(lam) = lam^2, lam_init = 1 (or higher on IC for time-irreversible PDEs), lr_w = 1e-3, lr_lam = 5e-3 to 1e-2.

## Results
On Allen-Cahn, viscous Burgers, Helmholtz, and 1-D wave equations, SA-PINN beats baseline PINN and learning-rate-annealing PINN by 1-3 orders of magnitude in L2 with fewer epochs (Allen-Cahn: ~2e-3 vs ~7e-2 baseline).
