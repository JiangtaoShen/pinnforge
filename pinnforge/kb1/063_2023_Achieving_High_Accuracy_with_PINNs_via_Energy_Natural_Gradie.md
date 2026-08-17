---
slot: 63
title: "Achieving High Accuracy with PINNs via Energy Natural Gradients"
authors: [Johannes Müller, Marius Zeinhofer]
year: 2023
venue: ICML 2023 (arXiv:2302.13163)
gitrepo: "https://github.com/MariusZeinhofer/Natural-Gradient-PINNs-ICML23"
---

## TL;DR
Standard PINN optimizers (GD, Adam, BFGS) plateau around relative L2 ~ 1e-3 because the squared-residual loss is ill-conditioned in parameter space. The **energy natural gradient** preconditions the gradient by the Gram matrix of parameter derivatives in function space; the resulting update is a Newton step in *function space* projected onto the model's tangent space. With a 1-D line search it reaches errors `1e-7..1e-9` — orders of magnitude better than Adam/BFGS.

## Problem
PINN loss `L(theta) = ||L u_theta - f||^2 + tau ||B u_theta - g||^2` is convex in the function `u`, but its parameter-space Hessian is highly anisotropic — squaring the differential operator squares the condition number. Adam/BFGS/Sobolev-natural-gradient cannot break the `1e-3..1e-4` accuracy floor.

## Method
Define the **energy Gram matrix** (Hessian of `E` in function-space evaluated on tangent directions):
$$
G_E(\theta)_{ij} = D^2 E(u_\theta)\big(\partial_{\theta_i} u_\theta,\, \partial_{\theta_j} u_\theta\big)
$$
For a quadratic PINN energy (linear PDE operator `L`):
$$
G_E(\theta)_{ij} = \int_\Omega \mathcal L(\partial_{\theta_i} u_\theta)\,\mathcal L(\partial_{\theta_j} u_\theta)\,dx + \tau\!\int_{\partial\Omega} \mathcal B(\partial_{\theta_i} u_\theta)\,\mathcal B(\partial_{\theta_j} u_\theta)\,ds
$$
The **energy natural gradient** is
$$
\nabla_E L(\theta) = G_E(\theta)^{+}\,\nabla L(\theta)
$$
Computed by solving the least-squares system `G_E(theta) psi = grad L(theta)` (use SVD-based `lstsq`, never explicit pseudo-inverse). Theorem 2: in function space, `DP_theta nabla_E L(theta) = Pi_{T_theta F_Theta}^{D^2 E}(D^2 E(u_theta)^{-1} grad E(u_theta))` — i.e. the parameter update equals the Newton step in function space projected onto the model tangent space. For PINN this approximately moves in the direction of the error `u* - u_theta`.

**Algorithm 1**:
1. Compute `g = grad_theta L(theta)`.
2. Build Jacobian `J in R^{N x p}` where row `i` is `[L u_theta(x_i)]_grad_theta` (and boundary rows for BC).
3. Form `G_E ≈ J^T J / N + tau J_bd^T J_bd / N_bd`.
4. Solve `G_E psi = g` by SVD lstsq.
5. Line-search `eta in [0, 1]` on a log grid; update `theta <- theta - eta* psi`.

Notes: Only a few hundred iterations are needed (small `p`); the cost per iter is `O(p N + p^3)` so the method is best with shallow networks (a few thousand parameters) — for deep networks restrict to last-layer or block-diagonal Gram. Activation must be smooth (`tanh`); double precision.

```python
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree

def jac_rows(params, apply_fn, X, op):
    """Per-row Jacobian d op(u_theta)(x) / d theta, returns [N, p]."""
    flat0, unravel = ravel_pytree(params)
    def op_flat(flat, x):
        return op(unravel(flat), apply_fn, x)              # scalar
    J = jax.vmap(lambda x: jax.grad(op_flat)(flat0, x))(X)  # [N, p]
    return J, flat0, unravel

def energy_ng_step(params, apply_fn, X_in, X_bd, f, g, tau=1.0,
                   line_search_grid=jnp.logspace(-3, 0, 12)):
    r_in = lambda p, ap, x: pde_operator(p, ap, x) - f(x)   # scalar per row
    r_bd = lambda p, ap, x: ap(p, x).squeeze() - g(x)

    J_in, flat0, unravel = jac_rows(params, apply_fn, X_in, r_in)
    J_bd, _, _          = jac_rows(params, apply_fn, X_bd, r_bd)
    r_in_v = jax.vmap(lambda x: r_in(params, apply_fn, x))(X_in)
    r_bd_v = jax.vmap(lambda x: r_bd(params, apply_fn, x))(X_bd)

    N_in, N_bd = X_in.shape[0], X_bd.shape[0]
    grad = (J_in.T @ r_in_v) / N_in + tau * (J_bd.T @ r_bd_v) / N_bd
    G_E  = (J_in.T @ J_in) / N_in + tau * (J_bd.T @ J_bd) / N_bd

    psi, *_ = jnp.linalg.lstsq(G_E, grad, rcond=None)

    def trial_loss(eta):
        new_params = unravel(flat0 - eta * psi)
        return pinn_loss(new_params, apply_fn, X_in, X_bd, f, g, tau)

    losses = jax.vmap(trial_loss)(line_search_grid)
    best   = line_search_grid[jnp.argmin(losses)]
    return unravel(flat0 - best * psi)
```

Hyper-params: shallow tanh net (1 hidden layer, 32-64 units suffices for 2-D Poisson), Gaussian init `sigma = 0.1`, fixed quadrature grid for the integrals, double precision (`jax.config.update("jax_enable_x64", True)`), ~500-2000 iterations.

## Results
On 2-D Poisson, 5-D Poisson, 1-D heat, and a nonlinear Deep Ritz problem, energy natural gradient reaches relative L2 of `1e-7..1e-9` after a few hundred iterations, while Adam (1e5 steps), BFGS, and Sobolev-natural-gradient stall at `1e-3..1e-4` — a 4-5 order-of-magnitude improvement.
