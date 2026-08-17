---
slot: 112
title: "General Explicit Network (GEN): A novel deep learning architecture for solving partial differential equations"
authors: [Genwei Ma, Ting Luo, Ping Yang, Xing Zhao]
year: 2026
venue: arXiv:2604.03321
gitrepo: ""
---

## TL;DR
Replace the point-to-point MLP in PINNs by a *point-to-function* architecture: every input coordinate `(x,t)` is first lifted to a vector of trainable analytic basis evaluations (`a_i sin(ω_i x + φ_i) + b_i` spatially, `α_j exp(-(t-μ_j)²/2σ_j²)` temporally), which a small MLP nonlinearly synthesises into `u(x,t)`. The architecture mimics a learnable truncated series expansion, embedding physical priors (e.g. d'Alembert for waves) into the basis itself, yielding far better extrapolation and robustness than tanh-MLP PINNs.

## Problem
Standard PINNs/MLPs learn a single closed-form-like approximation through compositions of `tanh`. They are pointwise, lose neighbourhood correlation, and extrapolate catastrophically outside the training domain. Series expansions (Fourier, power series) extrapolate stably because each basis function globally encodes structure - but their linear superposition is too rigid for nonlinear PDEs.

## Method

### A. Architecture: point-to-function synthesis
Spatial trigonometric bases `f_i(x) = a_i sin(ω_i x + φ_i) + b_i`, `i=1..m`; temporal Gaussian bases `g_j(t) = α_j exp(-(t-μ_j)²/2σ_j²)`, `j=1..n`. All of `(a_i, ω_i, φ_i, b_i, α_j, μ_j, σ_j)` are trainable. A small synthesis MLP `K_θ` (1 hidden layer, 20 tanh units) maps the concatenation `[f_1,...,f_m,g_1,...,g_n]∈R^{m+n}` to scalar `\hat u`:
$$
\hat u(x,t)=K_\theta\!\big(f_1(x),\ldots,f_m(x),\,g_1(t),\ldots,g_n(t)\big).
$$
This enables non-linear basis coupling (a strict generalisation of `Σ c_{ij} f_i g_j`).

### B. Physics-aware basis selection per PDE class
- Heat: only spatial trig/Gauss bases; rely on K_θ to learn the exp temporal decay.
- Wave: use characteristic coordinates `ξ_± = x ± ct` and split bases `φ_1(x-ct) + φ_2(x+ct)` to enforce d'Alembert duality and finite-speed propagation.
- Burgers / unknown structure: hybrid trig+Gauss in both x and t.

### C. Standard PINN loss
$$
\mathcal L=\mathbb E_{(x,t)\in\Omega}\big[\mathcal N[\hat u]\big]^2+\lambda\,\mathbb E_{\partial\Omega}[\hat u-u_{BC}]^2+\gamma\,\mathbb E_{t=0}[\hat u-u_0]^2.
$$

Init `a_i∼U(0,1)`, `ω_i∼iπ·U(0,1)`, `φ_i∼U(0,1)`, `μ_j∼[min,max]·U`, `σ_j∼U(0,1)`. `optax.adam` then optional L-BFGS.

```python
import jax, jax.numpy as jnp
import flax.linen as nn

class SinBasis(nn.Module):
    m: int

    @nn.compact
    def __call__(self, x):                              # x: (N,)
        a   = self.param("a",   lambda k: jax.random.uniform(k, (self.m,)))
        i   = jnp.arange(1, self.m+1, dtype=jnp.float32) * jnp.pi
        w   = self.param("w",   lambda k: i * jax.random.uniform(k, (self.m,)))
        phi = self.param("phi", lambda k: jax.random.uniform(k, (self.m,)))
        b   = self.param("b",   lambda k: jax.random.uniform(k, (self.m,)))
        return a * jnp.sin(w * x[:, None] + phi) + b

class GaussBasis(nn.Module):
    n: int
    tmin: float = 0.0
    tmax: float = 1.0

    @nn.compact
    def __call__(self, t):
        alpha = self.param("alpha", lambda k: jax.random.uniform(k, (self.n,)))
        mu    = self.param("mu",
                lambda k: self.tmin + (self.tmax-self.tmin)*jax.random.uniform(k, (self.n,)))
        sig   = self.param("sig",
                lambda k: 0.1 + 0.4*jax.random.uniform(k, (self.n,)))
        return alpha * jnp.exp(-((t[:, None] - mu)**2) / (2*sig**2))

class GEN(nn.Module):
    m: int = 16
    n: int = 16
    hidden: int = 20
    tmin: float = 0.0
    tmax: float = 1.0

    @nn.compact
    def __call__(self, x, t):
        fx = SinBasis(self.m)(x)
        gt = GaussBasis(self.n, self.tmin, self.tmax)(t)
        z  = jnp.concatenate([fx, gt], axis=-1)
        h  = nn.tanh(nn.Dense(self.hidden)(z))
        return nn.Dense(1)(h)

# wave equation: use ξ_± = x ± c t
class GEN_Wave(nn.Module):
    m: int = 16
    c: float = 1.0

    @nn.compact
    def __call__(self, x, t):
        f_minus = SinBasis(self.m, name="f_minus")(x - self.c * t)
        f_plus  = SinBasis(self.m, name="f_plus" )(x + self.c * t)
        h = nn.tanh(nn.Dense(20)(jnp.concatenate([f_minus, f_plus], axis=-1)))
        return nn.Dense(1)(h)
```

Hyper-parameters: `m=n=16-32`, hidden=20, `optax.adam(1e-3)`, IC/BC weights `γ,λ≈10`, batch ≈ 1024 collocation points.

## Results
GEN matches PINN accuracy *inside* the training interval on heat, wave, Burgers equations, and—critically—keeps reasonable accuracy when extrapolated *outside* the training domain (`|x|>1`, `t>T_train`) where MLP PINNs diverge. The learned spectral parameters (`ω_i`) align with the dominant Fourier components of the true solution, giving interpretability of the modes. Stable, robust solutions with fewer parameters than competing tanh-MLP / KAN baselines.
