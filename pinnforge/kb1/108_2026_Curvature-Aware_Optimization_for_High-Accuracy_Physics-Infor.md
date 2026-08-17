---
slot: 108
title: "Curvature-Aware Optimization for High-Accuracy Physics-Informed Neural Networks"
authors: [Anas Jnini, Elham Kiyani, Khemraj Shukla, Jorge F. Urbán, Nazanin Ahmadi Daryakenari, Johannes Müller, Marius Zeinhofer, George Em Karniadakis]
year: 2026
venue: arXiv:2604.05230
gitrepo: ""
---

## TL;DR
A practitioner's guide and benchmark for *curvature-aware* PINN optimizers: Self-Scaled BFGS (SSBFGS), Self-Scaled Broyden (SSBroyden), Natural Gradient (NG / Gauss-Newton on the residual), and Kronecker-eigenbasis SOAP. Self-scaling and Jacobi-preconditioned NG, combined with double precision and a damped-secant rule, reach 1e-6 to 1e-10 relative-L² accuracy on stiff/oscillatory/shock PDEs where Adam/L-BFGS stall.

## Problem
Vanilla PINNs are nonlinear least-squares `L(θ) = ½ ‖r(θ)‖²` with ill-conditioned NTK and severe spectral bias. First-order optimisers (Adam, AdamW) crawl on the resulting narrow ravines; vanilla BFGS / L-BFGS lose positive-definiteness from noisy gradients on stiff PDE residuals. The fix is curvature preconditioners that are also robust to PINN residual nonlinearity.

## Method

### A. Self-Scaled BFGS / Broyden
Update with secant pair `s_k=θ_{k+1}-θ_k`, `y_k=g_{k+1}-g_k`. Self-scaled family (`τ_k≠1`):
$$
H_{k+1}=\frac{1}{\tau_k}\Big[H_k-\frac{H_k y_k y_k^\top H_k}{y_k^\top H_k y_k}+\phi_k v_k v_k^\top\Big]+\frac{s_k s_k^\top}{y_k^\top s_k},\quad
v_k=(y_k^\top H_k y_k)^{1/2}\!\Big(\tfrac{s_k}{y_k^\top s_k}-\tfrac{H_k y_k}{y_k^\top H_k y_k}\Big).
$$
`(τ,φ)=(1,1)` → BFGS; `≠1` self-scales `H_k` to track Hessian magnitude. For batched PINN training, *skip the update* when `y_{k-1}^\top s_{k-1} < τ ‖s_{k-1}‖²` (damped secant) to retain positive-definiteness; with cubic line search.

### B. Natural Gradient / Gauss-Newton on residual
Cast loss as `L = ½ ‖r(θ)‖²`, Jacobian `J = ∂r/∂θ`. Update:
$$
\theta_{k+1}=\theta_k-\eta_k(J_k^\top J_k+\lambda I)^{-1}J_k^\top r(\theta_k).
$$
Implementations: low-rank when #residuals ≪ #params (solve `(J J^\top + λI)α = r`, take `θ←θ-η J^\top α`); Kronecker-factored (KFAC); randomized SVD. Jacobi scaling (diagonal preconditioner on `J^\top J`) stabilises early iterations.

### C. SOAP (Kronecker-eigenbasis Adam)
Per layer with reshaped gradient matrix `G`, accumulate `L=GG^\top, R=G^\top G`. Eigendecompose; run Adam-style updates in the rotated basis; rotate back. Diagonal in curvature eigenbasis combines structured curvature with diagonal adaptivity.

```python
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree

def cubic_line_search(loss_fn, theta, d, alpha0=1.0):
    # zoom-style cubic interpolation; details omitted
    ...

def ssbfgs_step(params, loss_fn, H, tau_min=1e-8):
    theta, unravel = ravel_pytree(params)
    g_pytree = jax.grad(loss_fn)(params)
    g = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(g_pytree)])
    d = -H @ g
    alpha = cubic_line_search(loss_fn, theta, d)
    theta_new = theta + alpha * d
    params_new = unravel(theta_new)
    g_new_pyt = jax.grad(loss_fn)(params_new)
    g_new = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(g_new_pyt)])
    s = alpha * d
    y = g_new - g
    ys = jnp.dot(y, s)
    if ys < tau_min * jnp.dot(s, s):                  # damped secant: skip
        return params_new, H
    rho = 1.0 / ys
    tau = jnp.dot(y, H @ y) / ys                      # self-scaling factor
    I = jnp.eye(H.shape[0])
    H_new = (I - rho*jnp.outer(s, y)) @ (H/tau) @ (I - rho*jnp.outer(y, s)) \
            + rho*jnp.outer(s, s)
    return params_new, H_new

def natural_gradient_step(params, residual_fn, x, lr=1e-2, lam=1e-6):
    flat, unravel = ravel_pytree(params)
    def r_of_flat(theta_flat):
        return residual_fn(unravel(theta_flat), x).reshape(-1)
    r = r_of_flat(flat)                               # (N_res,)
    J = jax.jacrev(r_of_flat)(flat)                   # (N_res, P)
    M = J @ J.T + lam * jnp.eye(J.shape[0])
    alpha = jnp.linalg.solve(M, r)
    update = J.T @ alpha
    flat_new = flat - lr * update
    return unravel(flat_new)
```

Hyper-parameters: double precision throughout (`jax.config.update("jax_enable_x64", True)`); SSBFGS / SSBroyden tolerance `τ≈1e-8`, cubic line search; NG damping `λ=1e-4..1e-6`, Jacobi diag-precondition, optional GMRES inner solve. Warm-start with `optax.adam(1e-3)` for 5-10k steps then switch to quasi-Newton.

## Results
Across Helmholtz 2-D/3-D (high wavenumber), Stokes, viscous Burgers, inviscid Burgers (with HLLC flux PINN), 1-D Euler (shock), and stiff PK-PD ODEs, SSBFGS / SSBroyden / NG reach 1e-6 to 1e-10 relative-L² where AdamW and Adam stall at 1e-2 to 1e-3. Self-scaling closes the gap to true Newton; the stochastic damped variant scales SSBFGS to multi-GPU batch training without losing accuracy.
