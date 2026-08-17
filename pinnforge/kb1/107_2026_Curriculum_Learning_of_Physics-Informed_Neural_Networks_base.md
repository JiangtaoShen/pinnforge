---
slot: 107
title: "Curriculum Learning of Physics-Informed Neural Networks based on Spatial Correlation"
authors: [Xujia Chen, Xinyue Hu, Letian Chen, Daming Shi, Wenhui Fan]
year: 2026
venue: arXiv:2605.15254
gitrepo: "https://github.com/pigofmomo/CurriculumLearningPINN"
---

## TL;DR
A spatial curriculum for BVP PINNs: layer the domain by distance from the boundary, ramp up causal weights from outside in, add a low-order polynomial "information bridge" that produces pseudo-labels to keep distant regions consistent, then finally rebalance per-subregion weights by joint loss-and-gradient magnitude. Together this fixes "no-propagation" / "incorrect-propagation" failures and low-frequency drift.

## Problem
Existing PINN curricula and causality-guided schemes order training in *time*. For boundary value problems (no explicit time direction) information must propagate from boundaries inward through spatial coupling. Single-shot global PINN training can converge locally near each boundary while remaining globally inconsistent (low-frequency drift) or fail to propagate information into the interior at all.

## Method

### A. Spatial layering + cumulative-exponential causal weights
Partition Ω into N subregions `{Ω_i}` and group into L layers indexed by integer distance `d_j` from `∂Ω`. The per-layer PDE loss is `L_PDE^{(j)} = (1/N_j) Σ_{x∈Λ_j} |N[u_θ](x)|²`. Outer-to-inner curriculum is enforced by exponentially decayed weights driven by accumulated outer-layer residuals:
$$
w_i=\exp\!\Big(-\epsilon\sum_{j=0}^{i-1}\mathcal L_{\text{PDE}}^{(j)}\Big),\qquad \mathcal L_{\text{PDE}}=\sum_{i=0}^{L-1}w_i\,\mathcal L_{\text{PDE}}^{(i)}.
$$
Inner layers stay suppressed until outer ones converge.

### B. Low-frequency information bridge (pseudo-labels)
Pick anchor points `A={x_i}` in interior/connector regions. Fit a low-order basis `φ(x)=[1,x,x²]^⊤` (1-D, higher degree in 2-D) to current predictions:
$$
\beta^\star=\arg\min_\beta \sum_i \alpha_i\big(\phi(x_i)^\top\beta - u_\theta(x_i)\big)^2,\quad \hat u_{LF}(x)=\phi(x)^\top\beta^\star.
$$
Add `L_LF = E_{x∼A}[u_θ(x)-\hat u_{LF}(x)]^2` to the loss. Bridge weight is largest in regions where curriculum weight `w_i` is still small.

### C. Gradient-aware region-adaptive reweighting (phase 2)
For each subregion compute `G_i = ‖∇_θ L_PDE^{(i)}‖_2`, score `s_i = log(L_PDE^{(i)}+ε) − λ_g log(G_i+ε)`, min-max normalise to `ρ_i ∈ [ρ_min,ρ_max]` (e.g. [1,5]) and use `L_PDE^{adapt} = (1/N) Σ ρ_i L_PDE^{(i)}`. Regions with high residual *and* small gradient (stuck) get amplified.

```python
import jax, jax.numpy as jnp

# --- Phase 1: spatial-causal layered loss ---
def layered_pde_loss(params, apply_fn, layers, pde_residual, eps=1.0):
    # layers: list of arrays, each = collocation set for layer j (outer -> inner)
    Ls = jnp.stack([jnp.mean(pde_residual(params, apply_fn, X)**2) for X in layers])
    acc = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(jax.lax.stop_gradient(Ls))[:-1]])
    w = jnp.exp(-eps * acc)
    return jnp.sum(w * Ls)

# --- Information bridge ---
def info_bridge(params, apply_fn, anchors, alpha, lam_LF=1.0, degree=2):
    u = apply_fn(params, anchors).squeeze(-1)
    Phi = jnp.stack([anchors[:, 0]**k for k in range(degree+1)], axis=1)
    W = jnp.diag(alpha)
    beta = jnp.linalg.lstsq(W @ Phi, W @ u, rcond=None)[0]
    u_LF = Phi @ beta
    return lam_LF * jnp.mean((u - u_LF)**2)

# --- Phase 2: region-adaptive reweighting ---
def region_adaptive_loss(params, apply_fn, regions, pde_residual,
                         lam_g=1.0, rho_min=1.0, rho_max=5.0, eps=1e-8):
    def Li_fn(p, X): return jnp.mean(pde_residual(p, apply_fn, X)**2)
    Ls = jnp.stack([Li_fn(params, X) for X in regions])
    def grad_norm(X):
        g = jax.grad(Li_fn)(params, X)
        return jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(g)) + eps)
    Gs = jnp.stack([grad_norm(X) for X in regions])
    s = jnp.log(jax.lax.stop_gradient(Ls) + eps) - lam_g*jnp.log(Gs + eps)
    s_tilde = (s - s.min()) / (s.max() - s.min() + eps)
    rho = rho_min + (rho_max - rho_min) * s_tilde
    return jnp.mean(rho * Ls)
```

Hyper-parameters: `ε∈[1,10]`, `λ_LF≈0.1–1.0` (decay over phase 1), bridge anchors ~10–30 (1D) / 50–200 (2D), 3 layers in 1-D, concentric layers in 2-D, `λ_g≈0.5`, `[ρ_min,ρ_max]=[1,5]`, `optax.adam(1e-3)` then L-BFGS for refinement.

## Results
On 1-D ODE and 2-D BVP benchmarks (including Helmholtz-type and reaction-diffusion stationary problems) the framework converts cases where vanilla / loss-balanced PINNs fail (zero solution, no-propagation, low-frequency drift) into accurate solutions, achieves 1-2 orders of magnitude lower relative L² error vs PINN, NTK-weighting, and prior subregion-reweighting baselines at comparable compute, with code at github.com/pigofmomo/CurriculumLearningPINN.
