---
slot: 113
title: "Goal oriented error estimation for adaptive sampling of PINNs"
authors: [Medard Govoeyi, Thomas Richter]
year: 2026
venue: arXiv:2604.01835
gitrepo: ""
---

## TL;DR
A fully mesh-free *Dual Weighted Residual* (DWR) error estimator for PINNs and Deep Ritz that measures the error in a user-chosen goal functional `J(u)` (e.g. point value, average), not the global L²/H¹ norm. Localise the estimator to drive adaptive sampling of collocation points toward regions that most influence `J(u)`, accelerating convergence of `J(u_θ)` orders of magnitude faster than residual-based RAR/RAD.

## Problem
Adaptive PINN sampling (RAR, RAD, RAR-D) uses the *pointwise PDE residual* as importance signal — good for global L² error, suboptimal for engineering quantities of interest (drag, point value, average over subdomain). Zienkiewicz-Zhu-style estimators were transplanted but need a background mesh, which defeats the mesh-free PINN advantage in high-D.

## Method
Consider Poisson `-Δu = f` in Ω, `u=g` on ∂Ω, approximated by Deep Ritz energy or strong-residual PINN over neural network set V_N. For a goal functional `J(u)`, introduce the *adjoint problem*
$$
\text{find } z\in V:\ \ a(\phi,z)=J(\phi)\quad\forall\phi\in V,
$$
where `a(u,v)=⟨∇u,∇v⟩_Ω + λ⟨u,v⟩_{∂Ω}` is the bilinear form of Deep Ritz (boundary penalty `λ`). The DWR error representation reads
$$
J(u)-J(u_\theta)\;\approx\;\rho(u_\theta)(z-z_\theta)=\langle f,z-z_\theta\rangle_\Omega-a(u_\theta,z-z_\theta),
$$
and is *localisable* to per-point indicators `η_i = ρ_i(u_θ)·(z-z_θ)(x_i)`.

### A. Pure-neural-network DWR
Train two PINNs in parallel: primal `u_θ` for the PDE and adjoint `z_φ` for the goal `J`. To get the *richer* dual weight `z - z_θ` without a mesh, the authors exploit the lack of linear-subspace structure of NN sets: train `z_φ` with *higher capacity* (more layers / different activation) than `u_θ`, or use a separate optimisation seed; the neural-network architecture itself provides the higher-order weight without needing finite-element enrichment.

### B. Localisation and adaptive sampling
Per collocation point `x_i` compute
$$
\eta_i=\Big|\big(f(x_i)+\Delta u_\theta(x_i)\big)\cdot\big(z_\varphi(x_i)-z_\theta(x_i)\big)\Big|.
$$
Then resample: keep top-K candidates by `η_i` (RAR-DWR) or sample new points with PDF `p(x)∝η(x)/Σ η` (RAD-DWR). Boundary points use `(u_θ-g)·(z_φ-z_θ)` analogously.

```python
import jax, jax.numpy as jnp

def primal_loss(params_u, apply_u, f_fn, g_fn, x_int, x_bd, lam=1e3, deep_ritz=True):
    if deep_ritz:
        u   = apply_u(params_u, x_int).squeeze(-1)
        gu  = jax.jacrev(lambda x: apply_u(params_u, x).squeeze(-1))(x_int)
        E   = 0.5*jnp.mean(jnp.sum(gu**2, axis=-1)) \
              - jnp.mean(f_fn(x_int) * u)
        ub  = apply_u(params_u, x_bd).squeeze(-1)
        return E + 0.5*lam*jnp.mean((ub - g_fn(x_bd))**2)
    else:
        ...                                              # strong PINN

def adjoint_loss(params_z, apply_z, J_fn, x_int, x_bd, lam=1e3):
    # Solves a(phi, z) = J(phi)  -> -Δz = J'(.), here J treated as source
    z   = apply_z(params_z, x_int).squeeze(-1)
    gz  = jax.jacrev(lambda x: apply_z(params_z, x).squeeze(-1))(x_int)
    E   = 0.5*jnp.mean(jnp.sum(gz**2, axis=-1)) - jnp.mean(J_fn(x_int) * z)
    zb  = apply_z(params_z, x_bd).squeeze(-1)
    return E + 0.5*lam*jnp.mean(zb**2)

def dwr_indicators(params_u, apply_u, params_z, apply_z, x_int, f_fn):
    # -Δu - f indicator weighted by (z_phi - z_theta) -- proxy z_theta = apply_u
    def u_of(x): return apply_u(params_u, x).squeeze(-1)
    def lap(x):
        H = jax.jacfwd(jax.grad(u_of))(x)               # per-point Hessian
        return jnp.trace(H)
    lap_u = jax.vmap(lap)(x_int)
    res   = f_fn(x_int) + lap_u                         # -Δu - f
    z_diff = apply_z(params_z, x_int).squeeze(-1) \
             - apply_u(params_u, x_int).squeeze(-1)
    return jnp.abs(res * z_diff)

def adaptive_resample(key, pool, eta, K, mode="rad"):
    eta = eta + 1e-12; p = eta / eta.sum()
    if mode == "rar":
        idx = jnp.argsort(-eta)[:K]
        return pool[idx]
    idx = jax.random.choice(key, pool.shape[0], shape=(K,), p=p)
    return pool[idx]
```

Hyper-parameters: alternate `optax.adam` updates on primal and adjoint (~200 steps each), boundary penalty `λ≈1e3`, resample every 1k steps, `K≈1000-4000`. Tanh / GeLU MLP, depth 4-6, width 50; the adjoint network uses 2× wider layers as the higher-capacity space.

## Results
On Laplace equations with localised goal functionals (point values, sub-region averages), DWR-adaptive sampling reaches a given functional accuracy `|J(u)-J(u_θ)|` with 5-20× fewer collocation points than uniform/RAR sampling for both Deep Ritz and strong-residual PINNs. Energy-functional minimisation also accelerates because the localiser concentrates effort where it matters for `J`.
