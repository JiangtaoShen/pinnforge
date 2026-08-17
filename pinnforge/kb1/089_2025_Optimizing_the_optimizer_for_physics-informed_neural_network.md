---
slot: 89
title: "Optimizing the Optimizer for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks"
authors: [Elham Kiyani, Khemraj Shukla, Jorge F. Urban, Jerome Darbon, George Em Karniadakis]
year: 2025
venue: "Computer Methods in Applied Mechanics and Engineering (arXiv:2501.16371)"
gitrepo: "https://github.com/EliKiani/Optimizing_the_Optimizer_PINNs"
---

## TL;DR
A systematic study showing **Self-Scaled BFGS (SSBFGS)** and **Self-Scaled Broyden (SSBroyden)** quasi-Newton methods - with proper line-search/trust-region - beat Adam and L-BFGS on PINN and PIKAN training by orders of magnitude on Burgers, Allen-Cahn, Kuramoto-Sivashinsky, Ginzburg-Landau, Stokes flow, and DeepONet operator learning. No adaptive weights, causal loss, or other tricks are needed.

## Problem
The PINN loss landscape with PDE residual is highly non-convex and stiff (visualisations show many local minima and (non)degenerate saddles). Adam zig-zags; L-BFGS gets stuck at saddles or terminates early under strong Wolfe. A *better-conditioned* quasi-Newton update can fix this.

## Method

### Broyden family + self-scaling
Quasi-Newton update (Broyden class):
$$ B_{k+1} = \tau_k\Big(B_k - \tfrac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k} + \theta_k (s_k^\top B_k s_k) w_k w_k^\top\Big) + \tfrac{y_k y_k^\top}{y_k^\top s_k} $$
where `s_k = x_{k+1}-x_k`, `y_k = grad f_{k+1} - grad f_k`, `theta_k=0` gives BFGS, `theta_k=1` gives DFP. **Self-scaling**: `tau_k = min(1, 1/b_k)` with `b_k = s_k^T B_k s_k / (y_k^T s_k)` (Al-Baali). For SSBroyden, `theta_k` is dynamically chosen so the scaled Hessian eigenvalues stay near 1. Line search uses Wolfe or Hager-Zhang; trust-region uses dogleg.

### Apply to PINN and PIKAN
Backbone: standard MLP PINN, or PIKAN with Chebyshev basis (KAN). Loss: standard PINN composite `L = L_ic + L_bc + L_r` *without* any adaptive weighting. Optimisation pipeline: short Adam warm-up (e.g. 1k steps), then run SSBroyden or SSBFGS until machine precision.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from jax.flatten_util import ravel_pytree

def value_and_flat_grad(loss_fn, params):
    v, grads = jax.value_and_grad(loss_fn)(params)
    g, unravel = ravel_pytree(grads)
    x, _ = ravel_pytree(params)
    return v, g, x, unravel

def strong_wolfe(phi, x, p, g, f0, c1=1e-4, c2=0.9, max_iter=25):
    """Returns alpha satisfying strong-Wolfe conditions; phi(a) returns (f, g'p)."""
    ...                                             # standard zoom line-search

def ssbroyden_step(loss_fn, params, state, cfg):
    """One SSBroyden step. state holds H (inverse Hessian approx, flat) and prev x, g."""
    f, g, x, unravel = value_and_flat_grad(loss_fn, params)
    if state.get("g_prev") is None:
        state.update(H=jnp.eye(x.size), x=x, g=g); return params, state, f
    s = (x - state["x"])[:, None]                   # (P, 1)
    y = (g - state["g"])[:, None]
    H = state["H"]
    rho = 1.0 / jnp.clip(y.T @ s, a_min=1e-12)
    b   = float((s.T @ (s / jnp.diag(H)[:, None])).squeeze())
    tau = min(1.0, 1.0 / max(b, 1e-12))             # Al-Baali self-scaling
    theta = cfg["theta"]
    # SSBroyden inverse update (eq. 8 in paper, condensed)
    Hy = H @ y
    H  = tau * (H - (Hy @ Hy.T) / (y.T @ Hy)) + rho * (s @ s.T)
    p_dir = -(H @ g[:, None]).squeeze(1)
    # strong-Wolfe line search
    def phi(a):
        new_params = unravel(x + a * p_dir)
        v, gn = jax.value_and_grad(loss_fn)(new_params)
        gn_flat, _ = ravel_pytree(gn)
        return float(v), float(jnp.dot(gn_flat, p_dir))
    alpha = strong_wolfe(phi, x, p_dir, g, f, c1=1e-4, c2=0.9, max_iter=25)
    new_x = x + alpha * p_dir
    new_params = unravel(new_x)
    state.update(H=H, x=new_x, g=g)
    return new_params, state, f

def closure(params):
    return loss_ic(params) + loss_bc(params) + loss_r(params)  # no adaptive weights

# Pipeline: Adam warm-up then SSBroyden
adam = optax.adam(1e-3); state_adam = adam.init(params)
for _ in range(1000):
    grads = jax.grad(closure)(params)
    upd, state_adam = adam.update(grads, state_adam)
    params = optax.apply_updates(params, upd)
state = {}; cfg = {"theta": 0.5}
for _ in range(50000):
    params, state, _ = ssbroyden_step(closure, params, state, cfg)
```

Hyperparameters: tanh MLP 4 x 50-128 (PINN) or PIKAN Chebyshev order 5; Adam warm-up `lr=1e-3` for 1k-10k steps; SSBroyden full-batch with strong-Wolfe line search (c1=1e-4, c2=0.9); enable `jax.config.update("jax_enable_x64", True)` for the last digits of accuracy. Variants: SSBFGS (`theta_k=0`), or SSBroyden with `theta_k` updated each step via Al-Baali's formula. Avoid stochastic batching for L-BFGS-type optimisers.

## Results
Burgers `nu=1e-2/pi`: SSBroyden reaches rel-L2 `~1e-9` (machine precision) vs `~1e-3` for L-BFGS and `~1e-2` for Adam. Allen-Cahn, Kuramoto-Sivashinsky, Ginzburg-Landau and lid-driven Stokes wedge show similar gaps (3-7 orders of magnitude). PIKANs with Chebyshev basis also benefit. DeepONet operator-learning loss is reduced by 1-2 orders. Demonstrates that *optimiser choice alone* outperforms most architectural / loss tricks in the PINN literature.
