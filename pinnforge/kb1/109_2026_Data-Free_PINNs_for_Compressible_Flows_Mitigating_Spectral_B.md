---
slot: 109
title: "Data-Free PINNs for Compressible Flows: Mitigating Spectral Bias and Gradient Pathologies via Mach-Guided Scaling and Hybrid Convolutions"
authors: [Ryosuke Yano]
year: 2026
venue: arXiv:2603.01001
gitrepo: ""
---

## TL;DR
A purely data-free PINN that captures detached bow shocks around a circular cylinder up to Ma=15 by (1) replacing MLP with a polar-grid radial-1D + azimuthal-2D anisotropic CNN with Fourier features, (2) Mach-number-dependent scaling of momentum / energy residuals that conditions a Hessian whose norm grows as O(Ma⁴), and (3) physics-anchored loss terms (upstream-fixing mask, stagnation analytical anchor, total-variation azimuthal regulariser, annealed artificial viscosity).

## Problem
Standard MLP PINNs cannot solve the steady 2-D compressible Euler equations around a blunt body without reference CFD data: (i) spectral bias smears shock discontinuities; (ii) Gauss-Newton Hessian norm scales as `‖H‖∼O(Ma⁴)` because residual `R∼O(Ma²)`, so AdamW and L-BFGS diverge; (iii) symmetric AD lets shock gradients leak upstream (Gibbs-like oscillations) and triggers the carbuncle instability at the stagnation point.

## Method

### A. Hybrid radial/azimuthal convolutional architecture on `(N_r×N_θ)` polar grid
Fourier-feature embedding of `(r,θ,x,y)`, then L=6 blocks each composed of (1) radial 1-D conv with large kernel `K_r=15` over the radial index, capturing long-range upstream-downstream coupling; (2) anisotropic 2-D conv with kernel `1×3` over azimuth, mimicking the `(1/r)∂_θ` operator; SiLU + LayerNorm + residual. Decoder MLP outputs `ρ, u, v, p` with Softplus positivity on `ρ` and `p`.

### B. Mach-guided dynamic residual scaling (regime-dependent)
For steady Euler residuals `R_mass, R_mom_{x,y}, R_energy`:
$$
\mathcal L_{\text{pde}}^{\text{A}}=\frac{1}{N}\sum\Big(|R_{\text{mass}}|^2+\frac{|R_{\text{mom}}|^2}{\mathrm{Ma}_\infty^2}+\frac{|R_{\text{energy}}|^2}{\mathrm{Ma}_\infty^4}\Big),\quad\mathrm{Ma}_\infty\ge 3
$$
$$
\mathcal L_{\text{pde}}^{\text{B}}=\frac{1}{N}\sum\Big(|R_{\text{mass}}|^2+\mathrm{Ma}_\infty^{2}|R_{\text{mom}}|^2+\mathrm{Ma}_\infty^{4}|R_{\text{energy}}|^2\Big),\quad\mathrm{Ma}_\infty\le 2
$$
Type A *pre-conditions* the stiff Hessian; Type B *amplifies* weak shocks to defeat spectral bias.

### C. Physics-anchored auxiliary losses
- Enthalpy `L_H = (H − H_∞)²` algebraic anchor;
- Upstream mask `L_mask` clamps `x < X_f(Ma)` to freestream;
- Stagnation anchor `L_stag` enforces analytical Rankine-Hugoniot ρ,p at `(-0.5,0)`;
- Slip-wall, symmetry, inflow Dirichlet/Neumann losses;
- Azimuthal TV `L_tv = Σ |ρ_{θ+1}−ρ_θ|` suppresses carbuncle;
- Artificial viscosity `R_mass ← ∇·(ρu)−ν Δρ` etc., with ν annealed `0.01→5e-4`.

Total: `L = L_pde + λ_H L_H + λ_nose L_nose + λ_in L_in + λ_wall L_wall + λ_stag L_stag + λ_sym L_sym + λ_mask L_mask + λ_tv L_tv`. Fixed static weights (boundary>>PDE): `λ_pde=1, λ_H=0.5, λ_in=20, λ_wall=500, λ_stag=200, λ_sym=20, λ_mask=1000, λ_tv=0.01` (0.1 at Ma=15).

Two-phase optimisation: `optax.adamw` (~20k epochs, ν annealed) → L-BFGS until float32 epsilon.

```python
import jax, jax.numpy as jnp
import flax.linen as nn

class HybridBlock(nn.Module):
    C: int = 64
    Kr: int = 15

    @nn.compact
    def __call__(self, z):                              # z: (B, Nr, Na, C)
        B, Nr, Na, C = z.shape
        # radial 1-D conv treats azimuth as independent batch axis
        zr = z.transpose(0, 2, 1, 3).reshape(B*Na, Nr, C)
        f  = nn.silu(nn.Conv(C, (self.Kr,), padding="SAME", name="conv_r")(zr))
        f  = f.reshape(B, Na, Nr, C).transpose(0, 2, 1, 3)
        h  = nn.LayerNorm(name="ln1")(z + nn.Dense(C, name="proj")(f))
        a  = nn.silu(nn.Conv(C, (1, 3), padding="SAME", name="conv_a")(h))
        return nn.LayerNorm(name="ln2")(h + a)

def mach_scaled_pde_loss(R_mass, R_mom_x, R_mom_y, R_e, Ma):
    if Ma >= 3:                                         # type A
        return jnp.mean(R_mass**2
                        + (R_mom_x**2 + R_mom_y**2)/Ma**2
                        + R_e**2 / Ma**4)
    else:                                               # type B
        return jnp.mean(R_mass**2
                        + (R_mom_x**2 + R_mom_y**2)*Ma**2
                        + R_e**2 * Ma**4)
```

## Results
Captures detached bow shock without CFD data for `2 ≤ Ma_∞ ≤ 15`; stand-off distance correctly decreases with Ma. Shock thickness is moderately wider than HLLC FVM CFD (due to artificial viscosity), but stagnation density / pressure match analytical Rankine-Hugoniot. Baseline MLP with identical setup completely fails (low-pass smearing). Wall-clock ~4 h on GTX 1070 vs ~30 min CFD; framework is justified as foundational for parametric/inverse extensions rather than steady-state acceleration.
