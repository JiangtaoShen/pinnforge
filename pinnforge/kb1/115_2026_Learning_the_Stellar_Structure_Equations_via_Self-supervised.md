---
slot: 115
title: "Learning the Stellar Structure Equations via Self-supervised Physics-Informed Neural Networks"
authors: [Manuel Ballester, Santiago Lopez-Tapia, Seth Gossage, Patrick Koller, Philipp M. Srivastava, Ugur Demir]
year: 2026
venue: arXiv:2604.06255
gitrepo: ""
---

## TL;DR
A self-supervised PINN solves the 4 stellar structure equations (hydrostatic + thermal equilibrium) on a Lagrangian mass coordinate `\hat M_r`, outputting continuous profiles `(\hat r, \hat P, \hat T, \hat L_r)`. Five carefully composed ingredients make it work: (1) hard-constrained boundary conditions via analytic ansatz, (2) SIREN backbone for multi-scale features, (3) auxiliary RFF-MLPs that smooth EoS / opacity tables into differentiable surrogates, (4) Stochastic-Projection PINN to dodge AD cost, (5) residual-based attention active sampling. Across MESA benchmark stars MRAE=3.06%, R²=0.9998.

## Problem
Stellar structure equations are stiff, multi-scale (solutions span many orders of magnitude), tightly coupled, and need exact boundary conditions at center *and* surface. Existing ML for stars is supervised on MESA outputs, hence biased. Naive PINNs fail because of the spectral bias, gradient pathologies, and the non-differentiable nature of the tabulated microphysics (EoS, opacity).

## Method

### A. Lagrangian formulation in `M̂_r`
Use enclosed mass as the independent variable (monotone, avoids the `r=0` singularity). Equilibrium system (after dropping `∂_t²r` and `∂_t S`):
$$
\frac{\partial\hat P}{\partial\hat M_r}=-\frac{\beta_a \hat M_r}{\hat r^4},\quad
\frac{\partial\hat r}{\partial\hat M_r}=\frac{\beta_b}{\hat r^2\hat\rho},\quad
\frac{\partial\hat T}{\partial\hat M_r}=-\frac{\beta_c \hat L_r}{\hat r^4}\nabla,\quad
\frac{\partial\hat L_r}{\partial\hat M_r}=\beta_d\,\varepsilon.
$$
Temperature gradient `∇` switches between radiative `∇_rad = 3κP̂ L̂_r/(16πacG M̂_r T̂^4)` and convective (Schwarzschild + mixing-length theory).

### B. Hard-constrained boundary conditions
Predict `y_θ` and post-process: `\hat r(\hat M_r) = \hat M_r · y_r,θ`, `\hat L_r = \hat M_r · y_{L,θ}` enforcing `r(0)=L_r(0)=0` by construction; analogous surface constraints match atmospheric BCs at `\hat M_r = \hat M_{tot}`.

### C. SIREN backbone for the main PINN
Sinusoidal activations: `h_{l+1}=sin(ω_0 W_l h_l + b_l)` with `ω_0≈30`. Captures multi-scale radial profiles in a compact model.

### D. Differentiable microphysics surrogates
Auxiliary MLPs with Random Fourier Feature (RFF) embeddings approximate EoS `\hat ρ(\hat P,\hat T,X,Y,Z)` and opacity `κ(\hat ρ,\hat T,X,Y,Z)` from MESA tables; pre-trained then frozen. Energy generation `ε` still computed via MESA's stiff finite-difference subroutine.

### E. SP-PINN: gradient-free derivatives + residual-based attention
Stochastic-Projection PINN approximates `∂_M_r u_θ` by Monte-Carlo finite differences against random unit projections, avoiding the AD cost of repeated higher-order derivatives. Active sampling uses a moving average of squared residual `s_i = ⟨r²(M_i)⟩` as the resampling weight per epoch (Residual-Based Attention).

### F. Composite loss
$$
\mathcal L=\mathcal L_{\text{PDE}}+\mathcal L_{\text{BC}}+\mathcal L_{\text{micro}},\quad
\mathcal L_{\text{PDE}}=\sum_{k=1}^4 \tfrac{1}{N_c}\sum_i w_i^{(k)} r_k^2(\hat M_i),
$$
with attention weights `w_i^{(k)}∝` residual EMA.

```python
import jax, jax.numpy as jnp
import flax.linen as nn

class SIRENPINN(nn.Module):
    hidden: int = 128
    depth: int = 5
    omega0: float = 30.0

    @nn.compact
    def __call__(self, Mhat):                           # Mhat: (N, 1)
        h = Mhat
        for i in range(self.depth):
            w_init = nn.initializers.uniform(scale=1.0/max(1, h.shape[-1]))
            h = nn.Dense(self.hidden, kernel_init=w_init, name=f"l{i}")(h)
            h = jnp.sin(self.omega0 * h) if i == 0 else jnp.sin(h)
        y = nn.Dense(4, name="head")(h)                 # (y_r, y_P, y_T, y_L)
        Mh = Mhat.squeeze(-1)
        r_hat = Mh * jnp.exp(y[:, 0])                   # r(0)=0
        L_hat = Mh * jnp.exp(y[:, 3])                   # L(0)=0
        P_hat = jnp.exp(y[:, 1])
        T_hat = jnp.exp(y[:, 2])
        return r_hat, P_hat, T_hat, L_hat

def stellar_residuals(params, apply_fn, Mhat, eos_apply, eos_p, kappa_apply, kappa_p,
                      eps_fn, betas, comp):
    def split_fn(Mh):
        return apply_fn(params, Mh)
    def d_dM(field_idx):
        # per-point gradient via vmap over jax.grad of scalar selector
        def scalar(Mh_scalar):
            return split_fn(jnp.array([[Mh_scalar]]))[field_idx][0]
        return jax.vmap(jax.grad(scalar))(Mhat.squeeze(-1))
    r, P, T, L = split_fn(Mhat)
    dr, dP, dT, dL = d_dM(0), d_dM(1), d_dM(2), d_dM(3)
    rho   = eos_apply(eos_p, P, T, *comp)               # smooth surrogate
    kappa = kappa_apply(kappa_p, rho, T, *comp)
    eps   = eps_fn(Mhat, T, rho, comp)                  # frozen MESA call
    nabla = compute_gradient(P, T, L, rho, kappa, Mhat, eos_apply, eos_p)
    R1 = dP + betas['a'] * Mhat.squeeze(-1) / r**4
    R2 = dr - betas['b'] / (r**2 * rho)
    R3 = dT + betas['c'] * L * nabla / r**4
    R4 = dL - betas['d'] * eps
    return R1, R2, R3, R4
```

Hyper-parameters: SIREN width 128, depth 4-6, `ω_0=30`; EoS / opacity MLPs 64×3 with RFF (`σ_RFF≈1.0`); `optax.adam(1e-3)` → L-BFGS; 5000 collocation points, RBA EMA momentum 0.99; SP-PINN 8 random projections per point.

## Results
Across benchmark MESA stars (multiple masses, single composition) the self-supervised PINN reproduces continuous radial profiles of `M_r, P, ρ, T, L_r` with MRAE 3.06% and average R² 99.98% — without using any MESA output as label during training. First demonstration of fully data-free PINN solving stellar structure with realistic microphysics; framework is differentiable end-to-end and ready for population-synthesis-scale (`>10^9` stars) inference and future time-dependent evolution.
