---
slot: 116
title: "Mapping-based Hard-constrained Physics-Informed Neural Networks for unbounded wave problems (MH-PINN)"
authors: [Tao Zhang, Hanshu Chen, Ilia Marchevsky, Zhuojia Fu]
year: 2026
venue: arXiv:2604.19843
gitrepo: ""
---

## TL;DR
For exterior Helmholtz / acoustic-scattering on `Ω = {r ≥ R_in}`, MH-PINN combines (i) an *algebraic* radial coordinate mapping `r(ξ)=R_in + L(1+ξ)/(1-ξ)` that compactifies `[R_in,∞)` to `ξ∈[-1,1)`, eliminating the unbounded-sampling and far-field truncation problem; (ii) an analytical ansatz `\hat u = Φ(x)·E(x)` where the *far-field asymptotic factor* `Φ(x)=e^{ik(|x|-R_in)}/|x|^{(d-1)/2}` enforces the Sommerfeld radiation condition by construction; (iii) a Dirichlet/Neumann hard constraint via an exact distance function `d(x)`. Loss reduces to PDE residual only — no soft BC/radiation penalties, no gradient competition.

## Problem
Standard PINNs fail on exterior Helmholtz problems: (1) cannot uniformly sample the infinite domain; (2) spectral bias prevents fitting high-frequency oscillations; (3) Sommerfeld radiation enforced softly as `λ_RAD‖√r(∂_r u − iku)‖²` is tiny compared to near-field PDE residual, causing gradient-magnitude imbalance that traps optimisation.

## Method

### A. Algebraic radial mapping `[R_in,∞) → [-1,1)`
$$
r(\xi)=R_{in}+L\,\frac{1+\xi}{1-\xi},\qquad
J(\xi)=\frac{dr}{d\xi}=\frac{2L}{(1-\xi)^2}.
$$
Differential operators are pulled back: `∂_r u = (1/J)∂_ξ u`, `∂_r^2 u = (1/J^2)∂_ξ^2 u - (J'/J^3) ∂_ξ u`. The 2-D Helmholtz becomes
$$
\frac{1}{J^2}\partial^2_\xi u-\frac{J'}{J^3}\partial_\xi u+\frac{1}{r(\xi)J}\partial_\xi u+\frac{1}{r(\xi)^2}\partial^2_\theta u + k^2 u = 0,
$$
solved on the bounded `(ξ,θ)∈[-1,1)×[0,2π)` so collocation points are finite and naturally concentrate near the inner boundary (near-field) while reaching infinity at `ξ=1`.

### B. Far-field asymptotic factor + neural envelope
$$
\hat u(x)=\Phi(x)\,E(x),\quad \Phi(x)=\frac{e^{ik(|x|-R_{in})}}{|x|^{(d-1)/2}},
$$
so `\hat u` analytically satisfies the Sommerfeld radiation condition `\lim_{r→∞} r^{1/2}(∂_r u - iku)=0` (in 2-D) for *any* envelope `E(x)`.

### C. Hard inner-boundary constraint via exact distance function
Build a normalised implicit distance function `d(x)` with `d|_Γ=0`, `∇d|_Γ = n`.
- Dirichlet `u|_Γ = g_D`:
$$
E(x)=\frac{g_D(x)}{\Phi(x)}+d(x)\,\mathcal N(\xi,\theta;\mathbf w).
$$
- Neumann `∂_n u|_Γ = g_N`, Taylor-shielded:
$$
E(x)=\mathcal N(\xi,\theta;\mathbf w)-d(x)\Big(\nabla\mathcal N\cdot\nabla d - h(x)\Big),\quad
h=\frac{g_N-\mathcal N \nabla\Phi\!\cdot\!\nabla d}{\Phi}.
$$

### D. Pure PDE loss
$$
\mathcal L=\frac{1}{N}\sum_i |f_{\text{PDE}}(\xi_i,\theta_i;\hat u)|^2.
$$
No BC and no radiation penalty terms.

```python
import jax, jax.numpy as jnp
import flax.linen as nn

class MHPINN(nn.Module):
    R_in: float = 1.0
    L: float = 1.0
    k: float = 5.0
    hidden: int = 64
    depth: int = 5
    dim: int = 2

    @nn.compact
    def __call__(self, xi, theta, d_fn, gD_fn=None, gN_fn=None):
        z = jnp.stack([xi, theta], axis=-1)
        h = z
        for _ in range(self.depth - 1):
            h = jnp.tanh(nn.Dense(self.hidden)(h))
        N = nn.Dense(2)(h)                              # real + imag
        N_c = N[..., 0] + 1j * N[..., 1]
        r   = self.R_in + self.L * (1 + xi) / (1 - xi + 1e-9)
        x   = r * jnp.cos(theta); y = r * jnp.sin(theta)
        Phi = jnp.exp(1j * self.k * (r - self.R_in)) / r**((self.dim - 1)/2)
        if gD_fn is not None:                           # Dirichlet
            E = gD_fn(x, y) / Phi + d_fn(x, y) * N_c
        else:                                           # Neumann (simplified)
            grad_d  = grad_of_d(d_fn, x, y)
            grad_N  = grad_of_N(N_c, xi, theta, r, theta)
            h_term  = (gN_fn(x, y) - N_c * grad_phi_dot_nd(Phi, grad_d)) / Phi
            E = N_c - d_fn(x, y) * (grad_N @ grad_d - h_term)
        return Phi * E                                  # complex

def helmholtz_residual_in_mapped_coords(params, apply_fn, xi, theta, k,
                                        d_fn, gD_fn):
    def u_of(xi_, th_):
        return apply_fn(params, jnp.array([xi_]), jnp.array([th_]),
                        d_fn, gD_fn=gD_fn)[0]
    # second derivs in xi and theta via nested jax.grad on real & imag parts
    u_xi   = jax.vmap(jax.grad(lambda x_, t_: u_of(x_, t_).real, 0))(xi, theta) \
           + 1j*jax.vmap(jax.grad(lambda x_, t_: u_of(x_, t_).imag, 0))(xi, theta)
    u_xixi = second_deriv_xi(u_of, xi, theta)
    u_tt   = second_deriv_theta(u_of, xi, theta)
    Lc = 1.0  # placeholder for self.L bound through closure
    J  = 2*Lc / (1 - xi)**2
    Jp = 4*Lc / (1 - xi)**3
    r  = 1.0 + Lc*(1 + xi)/(1 - xi)
    u  = jax.vmap(u_of)(xi, theta)
    return (u_xixi/J**2 - Jp/J**3*u_xi + (1/(r*J))*u_xi
            + (1/r**2)*u_tt + k**2 * u)
```

Hyper-parameters: `L≈R_in`, MLP 4×64 tanh, complex output split into real/imag, `optax.adam(1e-3)` → L-BFGS, ~5000 (ξ,θ) collocation points. `R_in` is just a background mapping radius; the actual scatterer shape lives in `d(x)`, allowing arbitrary non-circular boundaries.

## Results
On 2-D acoustic radiation from a unit-cylinder Dirichlet (`u(1,θ)=100 Pa`), standard PINN fails at `k≥5` (blurred fringes, spectral bias) while MH-PINN reproduces the Hankel-function analytical solution `u(r)=u_0 H_0^{(1)}(kr)/H_0^{(1)}(k)` cleanly up to `k=20` and on spatially varying `k(x)` (heterogeneous media). Sound-hard (Neumann) scattering from non-circular bodies works without re-tuning. Loss is purely PDE residual; no boundary/radiation hyper-parameters.
