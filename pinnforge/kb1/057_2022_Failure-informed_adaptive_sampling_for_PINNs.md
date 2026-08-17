---
slot: 057
title: "Failure-informed adaptive sampling for PINNs"
authors: [Zhiwei Gao, Liang Yan, Tao Zhou]
year: 2022
venue: SIAM J. Sci. Comput. (arXiv:2210.00279)
gitrepo: ""
---

## TL;DR
Treat residual-exceedance as a "failure event" from reliability analysis. Define a failure region `{x : |r(x;theta)| > eps_r}`, estimate its probability by **self-adaptive importance sampling** with a truncated Gaussian proposal, and add the failure samples to the collocation set. The training-set grows only where the PDE residual is large — far more sample-efficient than RAR or generative-model adaptive sampling.

## Problem
Fixed uniform collocation sets fail when the PDE solution has localized features (singularities, sharp fronts, unbounded domains). RAR draws a huge candidate pool and picks top-residual points (expensive in high dimensions). DAS trains a generative model per outer iteration (cost comparable to solving the PDE). A cheap, theory-backed adaptive sampler is needed.

## Method
Define a **limit-state function** `g(x) = |r(x;theta)| - eps_r` and the failure region `Omega_F = {g > 0}`. The failure probability under prior `omega(x)` is
$$
P_F = \int_\Omega \omega(x)\,\mathbb 1_{\Omega_F}(x)\,dx
$$
Importance sampling with proposal `h(x)` gives an unbiased estimator; the optimal (zero-variance) proposal `h_opt(x) ∝ 1_{Omega_F}(x) omega(x)` is unavailable, so SAIS approximates it by an iteratively refined **truncated Gaussian** `N_T(mu_k, Sigma_k)`:

1. Draw `N1` samples from `h_k`, sort by `g`, count `N_eta = #{g>0}`, set `N_p = p_0 N1` (e.g. `p_0=0.1`).
2. If `N_eta < N_p`: re-fit `mu_{k+1}, Sigma_{k+1}` from the top-`N_p` LSF samples (weighted by `omega`), set `h_{k+1} = N_T(mu_{k+1}, Sigma_{k+1})`. Repeat.
3. Else: terminate; the SAIS estimator is `P_F^SAIS = (1/N2) Σ (omega/h_opt) 1_{Omega_F}`; the new collocation points `D_adaptive` are the failure samples from the final draw.

**Outer FI-PINNs loop**: train PINN on `D_c ∪ D_adaptive`, estimate `P_F` via SAIS, stop if `P_F < eps_p`, else union new samples and retrain. Proven error bound `||u - u_theta||_{2,Omega} ≤ sqrt{2}/C1 * (S_Omega (M^2 eps_p + 2 eps_r) + 2 b)^{1/2}`.

```python
import jax, jax.numpy as jnp
import optax

def abs_residual(params, apply_fn, x):
    # scalar |r(x;theta)| for one point
    return jnp.abs(pde_operator(params, apply_fn, x))

def sample_gaussian(key, mu, L, n):              # L = chol(Sigma)
    z = jax.random.normal(key, (n, mu.shape[0]))
    return mu[None, :] + z @ L.T

def sais(key, params, apply_fn, d, eps_r,
         N1=10000, N2=10000, p0=0.1, M_iter=10):
    mu    = jnp.zeros(d)
    Sigma = jnp.eye(d)
    for k in range(M_iter):
        key, sub = jax.random.split(key)
        L = jnp.linalg.cholesky(Sigma + 1e-6 * jnp.eye(d))
        X = sample_gaussian(sub, mu, L, N1)
        g = jax.vmap(abs_residual, in_axes=(None, None, 0))(params, apply_fn, X) - eps_r
        Np    = int(p0 * N1)
        N_eta = int(jnp.sum(g > 0))
        if N_eta < Np:
            idx = jnp.argsort(-g)[:Np]
            top = X[idx]
            mu = jnp.mean(top, axis=0)
            Sigma = jnp.cov(top.T) + 1e-6 * jnp.eye(d)
        else:
            break
    key, sub = jax.random.split(key)
    L = jnp.linalg.cholesky(Sigma + 1e-6 * jnp.eye(d))
    X_final = sample_gaussian(sub, mu, L, N2)
    g_final = jax.vmap(abs_residual, in_axes=(None, None, 0))(params, apply_fn, X_final) - eps_r
    return X_final[g_final > 0]                  # new collocation points

# Outer FI-PINNs loop
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def inner_step(params, opt_state, D_c, D_b):
    def loss(p):
        return pde_loss(p, D_c) + lam_b * bc_loss(p, D_b)
    grads = jax.grad(loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

for outer in range(M_outer):
    for it in range(N_train_inner):
        params, opt_state = inner_step(params, opt_state, D_c, D_b)
    D_new = sais(key, params, apply_fn, d, eps_r)
    if D_new.shape[0] == 0:
        break
    D_c = jnp.concatenate([D_c, D_new], axis=0)
```

Hyper-params: `eps_r ≈ 1e-2..1e-1` (problem-dependent), `eps_p ≈ 1e-3`, `p_0 = 0.1`, `N_1 = N_2 ∈ [1e3, 1e4]`, M_outer ~ 5-20. Network: 3-4 hidden layers, 50 tanh units, Adam.

## Results
On a 2-D Poisson with peaked source, a singular elliptic problem, an unbounded-domain advection problem, and a time-dependent Burgers, FI-PINNs reduce relative L2 by 1-3 orders of magnitude vs uniform sampling and beat RAR at 5-10x fewer total residual evaluations because SAIS focuses sampling on the (small) failure region.
