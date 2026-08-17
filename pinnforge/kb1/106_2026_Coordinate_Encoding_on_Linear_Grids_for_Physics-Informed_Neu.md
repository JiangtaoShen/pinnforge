---
slot: 106
title: "Coordinate Encoding on Linear Grids for Physics-Informed Neural Networks (CELG)"
authors: [Tetsuro Tsuchino, Motoki Shiga]
year: 2026
venue: arXiv:2603.22700
gitrepo: ""
---

## TL;DR
Replace direct-coordinate input to a PINN with per-axis 1-D grids of trainable feature vectors, interpolated by a natural cubic spline (C²-continuous) and fused across axes by Hadamard product before an MLP head. This mitigates spectral bias like multi-resolution hashes but cuts space complexity from `O(M·R^D)` to `O(M·D·R)`, making it tractable in high dimension.

## Problem
Vanilla PINNs converge slowly because of spectral bias toward low frequencies. Grid-based encodings (Instant-NGP-style hash grids, PIXEL, Spline-PINN/H-Spline) help on 2-D/3-D problems but allocate features over the full `R^D` cell lattice — infeasible for high-D PDEs — and their interpolation kernels are only C⁰ (linear) or C¹ (cosine, Hermite). PINN losses require continuous *second* derivatives, so discontinuous spline derivatives at grid points cause unstable training and inaccurate predictions.

## Method
Place `R` grid points along each of the `D` coordinate axes. At every grid point `r` on axis `d` keep a trainable feature vector `g_d^{(r)} ∈ R^M`. For an arbitrary input `x = (x_1,...,x_D)`:

1. **Per-axis natural-cubic-spline interpolation** of `{g_d^{(r)}}_{r=1..R}` at `x_d` gives `z_d ∈ R^M`. Natural splines satisfy `z_d''(a_d) = z_d''(b_d) = 0` and are C² everywhere, so `∂²z_d/∂x_d²` is well-defined for the PDE residual.
2. **Multiplicative fusion** across axes: `φ = z_1 ⊙ z_2 ⊙ ··· ⊙ z_D` (Hadamard product; multiplicative interactions approximate richer functions than sums).
3. **MLP head** with `tanh` activations (smooth derivatives) outputs `u_θ(x) = MLP(φ)`.

The loss is the standard composite
$$
\ell(\theta)=\lambda_\text{pde}\ell_\text{pde}+\lambda_\text{init}\ell_\text{init}+\lambda_\text{bc}\ell_\text{bc}.
$$

Space complexity `O(M·D·R)`, time `O(M·D·R)` (tridiagonal solve for spline coefficients); vs PIXEL `O(M·R^D)` space and `O(M·2^D)` time.

```python
import jax, jax.numpy as jnp
import flax.linen as nn

class NaturalSpline1D(nn.Module):
    """Natural cubic spline with trainable feature values g at R fixed knots."""
    R: int
    M: int
    a: float
    b: float

    @nn.compact
    def __call__(self, x):                               # x: (N,)
        knots = jnp.linspace(self.a, self.b, self.R)
        g = self.param("g", lambda k: 0.1*jax.random.normal(k, (self.R, self.M)))
        h = knots[1:] - knots[:-1]                       # (R-1,)
        # natural BC: m_0 = m_{R-1} = 0; solve tridiagonal for inner m
        m = solve_natural_spline(knots, g)               # (R, M)  (Thomas O(R))
        idx = jnp.clip(jnp.searchsorted(knots, x) - 1, 0, len(h) - 1)
        x_l = knots[idx]; x_r = knots[idx + 1]
        hi  = h[idx]
        A = (x_r - x) / hi; B = (x - x_l) / hi
        C = (A**3 - A) * hi**2 / 6.0; D = (B**3 - B) * hi**2 / 6.0
        return (A[:,None]*g[idx] + B[:,None]*g[idx+1]
                + C[:,None]*m[idx] + D[:,None]*m[idx+1])

class CELG(nn.Module):
    D: int
    R: int
    M: int
    hidden: int = 64
    depth: int = 4
    bounds: tuple = None                                 # ((a0,b0), ..., (aD,bD))

    @nn.compact
    def __call__(self, x):                               # x: (N, D)
        z = NaturalSpline1D(self.R, self.M, *self.bounds[0], name="sp0")(x[:, 0])
        for d in range(1, self.D):
            zd = NaturalSpline1D(self.R, self.M, *self.bounds[d],
                                 name=f"sp{d}")(x[:, d])
            z = z * zd                                   # Hadamard
        h = nn.tanh(nn.Dense(self.hidden)(z))
        for _ in range(self.depth - 1):
            h = nn.tanh(nn.Dense(self.hidden)(h))
        return nn.Dense(1)(h)
```

Hyper-parameters: `R≈32–64` per axis, `M≈16–32`, MLP hidden 64 with depth 3-4 (tanh), `optax.adam(1e-3)`, batch ≈ a few thousand collocation points. Higher-order PDEs need higher-order natural splines.

## Results
On multi-band Poisson (up to D=10), Burgers, Allen–Cahn and flow-mixing, CELG matches or beats PINN, PIXEL, H-Spline (Spline-PINN) and tensor-CP variants in relative L² while training much faster (×5–10 fewer epochs to target accuracy) and at small fixed memory in high D where PIXEL/H-Spline cannot allocate the grid. The natural-spline C² continuity ablation confirms training divergence with linear / cosine / Hermite interpolation as derivatives blow up at grid points.
