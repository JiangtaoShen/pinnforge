---
slot: 021
title: "Optimally weighted loss functions for solving PDEs with Neural Networks"
authors: [Remco van der Meer, Cornelis Oosterlee, Anastasia Borovykh]
year: 2020
venue: "J. Comp. Appl. Math. (arXiv:2002.06269)"
gitrepo: "https://github.com/remcovandermeer/Optimally-Weighted-PINNs"
---

## TL;DR
Derives a closed-form optimal loss weight lambda between residual and BC terms by minimizing an epsilon-closeness bound for linear well-posed PDEs, then turns it into a self-supervised heuristic ("Magnitude Normalization") that needs no access to the true solution. Plug-in replacement for the L = L_r + L_b weighting.

## Problem
Vanilla PINN loss L = L_I + L_B implicitly assumes the residual and boundary terms have comparable scales. They do not: for a 2-D Laplace eigenproblem with frequency omega the theoretically optimal lambda ranges from ~1.6e-2 (omega=pi) to ~1.6e-5 (omega=10 pi). Wrong scaling makes well-posed problems unsolvable with vanilla PINN.

## Method
Generalize the loss to a convex combination
$$
\mathcal{L}(\hat u) = \lambda\,L_I(\hat u) + (1-\lambda)\,L_B(\hat u)
$$
with L_I = mean |N(x,u_hat) - F|^p over interior, L_B = mean |B(x,u_hat) - G|^p over boundary (p=2 in practice).

**A. Optimal lambda (needs true solution u).** Define magnitude bounds
$$
M_I(u) = \int_\Omega |N(x,u)|^p\,dx, \quad M_B(u) = \int_{\partial\Omega} |B(x,u)|^p\,dx_\Gamma
$$
The min-max derivation gives
$$
\lambda^* = \frac{M_B(u)}{M_I(u) + M_B(u)}
$$
This choice makes the ratio lambda L_I / (1-lambda) L_B scale-invariant under c1*N=0, c2*B=0 rewrites.

**B. Magnitude Normalization (practical heuristic).** Replace u by the current network u_hat — lambda is no longer a constant hyperparameter, it is a stop-gradient functional of the prediction:
$$
\mathcal{L}(\hat u) = \frac{M_B(\hat u)\,L_I(\hat u) + M_I(\hat u)\,L_B(\hat u)}{M_I(\hat u) + M_B(\hat u)}
$$
M_I(u_hat) and M_B(u_hat) are evaluated on the current network with autograd, detached from the graph.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

def residual_op(params, apply_fn, x_int):
    # For Poisson: N[u] = Laplacian(u). Compute via Hessian trace.
    def u_fn(x): return apply_fn(params, x)[0]
    def N_at(x):
        H = jax.hessian(u_fn)(x)
        return jnp.trace(H) - F(x)
    return jax.vmap(N_at)(x_int)

def boundary_op(params, apply_fn, x_bnd):
    return jax.vmap(lambda x: apply_fn(params, x)[0] - G(x))(x_bnd)

def magnitude_normalized_loss(params, apply_fn, x_int, x_bnd, p=2):
    Nr = residual_op(params, apply_fn, x_int)
    Bb = boundary_op(params, apply_fn, x_bnd)
    LI = jnp.mean(jnp.abs(Nr)**p)
    LB = jnp.mean(jnp.abs(Bb)**p)
    # detach so lambda does not back-propagate
    MI = jax.lax.stop_gradient(LI)
    MB = jax.lax.stop_gradient(LB)
    denom = MI + MB + 1e-12
    return (MB * LI + MI * LB) / denom

# Optimization: Adam warmup, then L-BFGS via optax / jaxopt.
opt = optax.adam(1e-3)
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, x_int, x_bnd):
    grads = jax.grad(lambda p:
        magnitude_normalized_loss(p, apply_fn, x_int, x_bnd))(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
# Follow-up: jaxopt.LBFGS(fun=...).run(params) for fine convergence.
```

Adaptive collocation: start with few points (e.g. 2 interior + 2 boundary), double when L-BFGS stalls. Recommended: tanh MLP, L-BFGS, p=2.

## Results
On 2-D Laplace eigenproblems with omega up to 10 pi the original PINN fails for omega > 4 pi; optimal-weight and Magnitude-Normalization variants stay accurate, often 2-3 orders of magnitude lower L2/L-inf error. Works in high-dimensional PDEs as well.
