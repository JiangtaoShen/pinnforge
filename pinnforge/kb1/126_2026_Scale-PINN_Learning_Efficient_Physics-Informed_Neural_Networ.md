---
slot: 126
title: "Scale-PINN: Learning Efficient Physics-Informed Neural Networks Through Sequential Correction"
authors: [Pao-Hsiung Chiu, Jian Cheng Wong, Chin Chun Ooi, Chang Wei, Yuchen Fan, Yew-Soon Ong]
year: 2026
venue: arXiv:2602.19475
gitrepo: "https://github.com/chiuph/SCALE-PINN"
---

## TL;DR
Scale-PINN imports the iterative-residual-correction principle from classical numerical solvers into the PDE loss. An auxiliary "sequential correction" term `(1/τ_sc)·B(f(·;w^k) − f(·;w^{k-1}))` augments the standard residual; choosing `B = P_α = (I − α²∇²)` (implicit residual smoothing / modified Richardson iteration) gives stable, fast convergence with plain SGD/Adam. Lid-driven cavity at Re=3200 trains in ~90 s vs 12–15 h for PirateNets/SOAP; competitive with high-fidelity CFD. (Reference implementation is in JAX.)

## Problem
Vanilla PINNs have rugged loss landscapes — multiple local minima, oscillatory paths, premature convergence on stiff/multiscale problems. To stabilise, papers resort to large batches (slow), curriculum learning (manual), or second-order optimizers (expensive). Lid-driven cavity at Re ≥ 3200 typically needs 10-15 h. The standard PDE loss lacks the residual-correction structure that classical iterative solvers exploit.

## Method

### A. Generic sequential-correction loss
For iterative scheme `u^{k+1} = u^k + B^{-1} r^k` with `r^k = h − A u^k`, Taylor extrapolation `u^{k+1} = 2u^k − u^{k-1}` yields
$$L^k_{\text{sc-pde}} = \bigl\|N_\vartheta[f(\cdot;w^k)] - h(\cdot) + \tfrac{1}{\tau_{sc}} \mathbb{B}\bigl(f(\cdot;w^k) - f(\cdot;w^{k-1})\bigr)\bigr\|_{L^2}^2$$
`τ_sc>0` is the relaxation hyperparameter; `B=0` recovers standard PINN.

### B. Residual-smoothing operator `P_α = (I − α²∇²)`
Choose `B = P_α` (implicit residual smoothing):
$$L^k_{\text{sc-pde}} = \bigl\|N_\vartheta[f^k] - h + (M_f - M_v)\bigr\|^2$$
with `M_f = (1/τ_sc) f^k − (γ/τ_α) ∇² f^k`, `M_v = (1/τ_sc) f^{k-1} − (γ/τ_α) ∇² f^{k-1}`, `α² = τ_sc γ/τ_α`. Tunable `τ_sc, γ, τ_α > 0`.

### C. Continuity equation patch (incompressible NS)
For Navier-Stokes, decouple pressure-velocity via an auxiliary term in the continuity equation:
$$L^k_{\text{sc-pde}}(\text{Cn}) = \bigl\|\partial_x u^k + \partial_y v^k + (1/\tau_{sc})(p^k - p^{k-1})\bigr\|^2$$
That makes the pressure dynamically respond to the divergence — a standard pressure-Poisson trick repurposed as a PINN-loss term.

### D. Training is just SGD/Adam with stored previous-iteration network
At each step store `w^{k-1}`, forward-pass to evaluate `f^{k-1}` and `∇² f^{k-1}` on the current mini-batch, then proceed with the modified loss. Adds 1 forward + 2 backward passes per step but no new architecture, no curriculum, no pretraining. **Network**: MLP with `sin(F·π·)` on layer 1 only (frequency annealing) and SiLU on layers 2..L.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    """First layer: sin(F·π··) frequency annealing; layers 2..L: SiLU."""
    out_dim: int
    hidden: int = 128
    depth: int = 5
    freq: float = 1.0
    @nn.compact
    def __call__(self, x):
        h = jnp.sin(self.freq * jnp.pi * nn.Dense(self.hidden)(x))   # layer 1
        for _ in range(self.depth - 1):
            h = nn.silu(nn.Dense(self.hidden)(h))                    # layers 2..L
        return nn.Dense(self.out_dim)(h)

def laplacian(u_fn, x):                                       # Σ_k ∂²/∂x_k² of scalar u_fn
    hess = jax.hessian(u_fn)(x)                               # (d, d)
    return jnp.trace(hess)

def scale_pinn_step(params, params_prev, opt_state, X_int, X_bc,
                    tau_sc=0.1, gamma=1.0, tau_alpha=1.0, lam_bc=10.0):
    def loss_fn(p):
        u_fn  = lambda x: net.apply(p, x[None]).squeeze()
        u_pfn = lambda x: net.apply(params_prev, x[None]).squeeze()
        u_k    = jax.vmap(u_fn )(X_int)
        u_km1  = jax.vmap(u_pfn)(X_int)
        L_k    = jax.vmap(lambda x: laplacian(u_fn,  x))(X_int)
        L_km1  = jax.lax.stop_gradient(
                 jax.vmap(lambda x: laplacian(u_pfn, x))(X_int))
        M_f = u_k   / tau_sc - (gamma / tau_alpha) * L_k
        M_v = u_km1 / tau_sc - (gamma / tau_alpha) * L_km1
        pde_res = pde_residual_fn(p, X_int)                   # N_ϑ[u^k] − h
        res = pde_res + (M_f - M_v)
        L_pde = jnp.mean(res ** 2)
        L_bc  = bc_loss(p, X_bc)
        return L_pde + lam_bc * L_bc
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss

net = MLP(out_dim=3)
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
params_prev = jax.tree_util.tree_map(jnp.copy, params)
opt = optax.adam(1e-3); opt_state = opt.init(params)

for it in range(50_000):
    params, opt_state, L = scale_pinn_step(params, params_prev, opt_state,
                                           X_int, X_bc)
    params_prev = jax.tree_util.tree_map(jnp.copy, params)     # snapshot for next step
```

Hyperparameters (lid-driven cavity): MLP with a 128-wide sinusoidal first layer + 32-wide shared/per-variable (u,v,p) branches (256/64-wide for Re≥7500), sin(F·π·) on layer 1 + SiLU thereafter, He init; Adam with warm-up cosine decay (min lr 1e-10); batch 400-2400 / iter; 50k-100k iters; `γ_uv = 1/Re`. No pretraining.

## Results
Lid-driven cavity Re=400: relative L2 1.43e-2 in 90 s (vs 1800 s vanilla PINN). Re=3200: 1.73e-2 in 90 s (PirateNets/SOAP need 12-15 h). Re=7500/10k/20k: training scales 150-380 s with error 2.97e-2-4.43e-2. Aerodynamic flow past airfoils (Re=500-1000), flow past staggered squares (Re=25), Rayleigh-Bénard at Ra=100k all converge in 180-390 s with sub-2% relative error. KS, Grey-Scott, KdV, Allen-Cahn benchmarks all solved within 10 min with state-of-the-art accuracy.
