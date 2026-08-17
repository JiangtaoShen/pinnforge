---
slot: 026
title: "When and why PINNs fail to train: A neural tangent kernel perspective"
authors: [Sifan Wang, Xinling Yu, Paris Perdikaris]
year: 2020
venue: "J. Comp. Phys. (arXiv:2007.14527)"
gitrepo: "https://github.com/PredictiveIntelligenceLab/PINNsNTK"
---

## TL;DR
Derive the limiting NTK of PINNs and show K = block-diag(K_uu, K_rr): the residual block K_rr typically has eigenvalues 10^2-10^3 larger than the BC/IC block K_uu, so the residual term converges fast and the BC term converges slowly — the cause of training failure. Fix: rescale loss weights every step by NTK trace ratios.

## Problem
Two empirical pathologies of vanilla PINN — spectral bias and stalled BC fitting — were unexplained. NTK analysis reveals (i) eigenvalues of K decay rapidly (spectral bias) and (ii) Tr(K_rr) >> Tr(K_uu), so the BC term has a much smaller learning-rate-equivalent and never catches up.

## Method
NTK of PINN with loss L = lambda_b L_b + lambda_r L_r is a block matrix
$$
K(t) = \begin{bmatrix} K_{uu}(t) & K_{ur}(t)\\ K_{ru}(t) & K_{rr}(t)\end{bmatrix},\quad
K_{uu}^{ij} = \langle\partial_\theta u(x_b^i),\partial_\theta u(x_b^j)\rangle
$$
$$
K_{rr}^{ij} = \langle\partial_\theta\mathcal{L}u(x_r^i),\partial_\theta\mathcal{L}u(x_r^j)\rangle
$$
Define average convergence rate c = Tr(K)/n. Algorithm 1 sets at each step (or every 10 steps):
$$
\lambda_b = \frac{\mathrm{Tr}(K)}{\mathrm{Tr}(K_{uu})},\quad
\lambda_r = \frac{\mathrm{Tr}(K)}{\mathrm{Tr}(K_{rr})}
$$
Tr(K_uu) and Tr(K_rr) are cheap to estimate by power iteration or by ||J_u||_F^2 and ||J_r||_F^2, the squared Frobenius of the Jacobian rows.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

def per_sample_bc(params, apply_fn, x):
    return apply_fn(params, x)[0] - g_b(x)

def per_sample_pde(params, apply_fn, x):
    return pde_op(params, apply_fn, x) - f_r(x)

def ntk_trace(per_sample_fn, params, xs):
    """Tr(K) = sum_i ||d r_i / d theta||^2.  Uses jacrev + tree reduction."""
    # Per-sample gradient via vmap of grad.
    grad_one = jax.grad(lambda p, x: per_sample_fn(p, apply_fn, x))
    grads = jax.vmap(grad_one, in_axes=(None, 0))(params, xs)  # pytree, leading N
    leaves = jax.tree_util.tree_leaves(grads)
    sqs = [jnp.sum(l**2) for l in leaves]   # sums over (N, *param_shape) -> scalar
    return sum(sqs)

def loss_b(params, apply_fn, x_b):
    r = jax.vmap(lambda x: per_sample_bc(params, apply_fn, x))(x_b)
    return 0.5 * jnp.mean(r**2)

def loss_r(params, apply_fn, x_r):
    r = jax.vmap(lambda x: per_sample_pde(params, apply_fn, x))(x_r)
    return 0.5 * jnp.mean(r**2)

opt = optax.sgd(1e-4); state = opt.init(params)
lam_b, lam_r = 1.0, 1.0

@jax.jit
def grad_step(params, state, lam_b, lam_r, x_b, x_r):
    def total(p):
        return lam_b * loss_b(p, apply_fn, x_b) + lam_r * loss_r(p, apply_fn, x_r)
    g = jax.grad(total)(params)
    upd, state = opt.update(g, state, params)
    return optax.apply_updates(params, upd), state

for step in range(N):
    if step % 10 == 0:
        Tr_uu = ntk_trace(per_sample_bc,  params, x_b)
        Tr_rr = ntk_trace(per_sample_pde, params, x_r)
        Tr_K  = Tr_uu + Tr_rr
        lam_b = Tr_K / jnp.maximum(Tr_uu, 1e-12)
        lam_r = Tr_K / jnp.maximum(Tr_rr, 1e-12)
    params, state = grad_step(params, state, lam_b, lam_r, x_b, x_r)
```

Recommended: tanh, NTK parameterization (1/sqrt(d_h) scaling on weights), full-batch SGD or Adam, update lambdas every 10 steps. Use Hutchinson estimator for trace if mini-batched.

## Results
On 1-D Poisson with sin(a pi x), a=1,2,4 the eigenvalue plots show Tr(K_rr) is 10^2-10^4 larger than Tr(K_uu). Algorithm 1 cuts relative L2 error by ~2 orders of magnitude vs vanilla PINNs (≈150x on 1-D Poisson, ≈260x on the 1-D wave equation), the paper's two benchmarks.

<!-- input quality issue: pymupdf-fallback markdown — equations rendered as broken LaTeX -->
