---
slot: 76
title: "Kolmogorov-Arnold-Informed neural network (KINN): A physics-informed deep learning framework for solving forward and inverse problems based on Kolmogorov-Arnold Networks"
authors: [Yizheng Wang, Jia Sun, Jinshuai Bai, Cosmin Anitescu, Mohammad Sadegh Eshaghi, Xiaoying Zhuang, Timon Rabczuk, Yinghua Liu]
year: 2024
venue: "Computer Methods in Applied Mechanics and Engineering (arXiv:2406.11045)"
gitrepo: "https://github.com/yizheng-wang/Research-on-Solving-Partial-Differential-Equations-of-Solid-Mechanics-Based-on-PINN"
---

<!-- input is pymupdf-fallback plain text -->

## TL;DR
KINN replaces the MLP backbone in three PDE formulations - strong-form (PINN), energy-form (DEM) and inverse boundary-integral form (BINN) - with a B-spline Kolmogorov-Arnold Network (KAN). KAN's learnable univariate edge activations match the basis-function nature of numerical PDE solvers and outperform MLPs on multi-scale, singular, stress-concentration, hyperelastic and heterogeneous problems with far fewer parameters; MLPs remain better on complex geometries.

## Problem
MLP-based PINN/DEM/BINN suffer from spectral bias, slow convergence near singularities and stress concentrations, and excessive parameter counts. The Kolmogorov-Arnold representation theorem provides an alternative decomposition where each edge carries a learnable 1-D function - well-aligned with B-spline/FEM-type interpolation.

## Method
A KAN layer `[l_i, l_o]` replaces a Linear+activation by a sum over learnable univariate splines on each edge:
$$ y_i = \sum_{j=1}^{l_i} \phi_{ij}(x_j),\quad \phi_{ij}(x) = w_s\, \text{silu}(x) + w_b \sum_{m=1}^{G+k} c_m^{(ij)} B_m(x) $$
with `G` grid intervals and B-spline order `k` (default `k=3`, `G=5-20`). Three KINN variants share this backbone:

- **KINN_PINN** (strong form): `L = lambda_r/N_r sum |P(u) - f|^2 + lambda_b/N_b sum |B(u) - g|^2`.
- **KINN_DEM** (energy form): minimise discretised potential `L = 1/2 int_Omega grad(u).grad(u) dOmega - int_Gamma_t t_bar u dGamma - int_Omega f u dOmega` with hard Dirichlet via `u = u_p(x;theta_p) + D(x) u_g(x;theta_g)` (distance network).
- **KINN_BINN** (inverse / boundary-integral form): residual built from the fundamental solution `u_f(x;y) = -ln(r)/(2 pi)` with piece-wise singular-integral regularisation.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class KANLayer(nn.Module):
    in_dim: int
    out_dim: int
    G: int = 5
    k: int = 3
    lo: float = -1.0
    hi: float = 1.0

    @nn.compact
    def __call__(self, x):                          # x: (B, in_dim)
        grid = jnp.linspace(self.lo, self.hi, self.G + 1)
        coef = self.param("coef", nn.initializers.normal(0.1),
                          (self.in_dim, self.out_dim, self.G + self.k))
        w_s  = self.param("w_s", nn.initializers.ones, (self.in_dim, self.out_dim))
        w_b  = self.param("w_b", nn.initializers.ones, (self.in_dim, self.out_dim))
        Bx   = b_spline_basis(x, grid, self.k)      # (B, in_dim, G+k); Cox-de Boor
        spline = jnp.einsum("bik,iok->bio", Bx, coef)
        base   = nn.silu(x)[..., None]              # (B, in_dim, 1)
        return (w_s * base + w_b * spline).sum(axis=1)  # (B, out_dim)

class KAN(nn.Module):
    shape: tuple = (2, 5, 5, 1)
    G: int = 5
    k: int = 3
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.shape) - 1):
            x = KANLayer(self.shape[i], self.shape[i+1], G=self.G, k=self.k)(x)
        return x

# Strong-form loss (KINN_PINN)
def loss_pinn(params, x_r, x_b, f, g, lam_r=1.0, lam_b=1.0):
    u_at = lambda xi: KAN().apply(params, xi[None])[0, 0]
    laplacian = lambda xi: jnp.trace(jax.hessian(u_at)(xi))
    Pu = jax.vmap(laplacian)(x_r) + ...             # operator P
    u_b = jax.vmap(u_at)(x_b)
    L_r = jnp.mean((Pu - f) ** 2)
    L_b = jnp.mean((u_b - g) ** 2)
    return lam_r * L_r + lam_b * L_b
```

For DEM, integrate with Gauss/Simpson quadrature on a structured grid; for BINN integrate over boundary panels with singular-integral regularisation. Train with `optax.adam(1e-3)` then `optax.lbfgs()`. KAN grids can be extended periodically (re-init coefficients by least squares).

## Results
KINN (especially KINN_DEM) clearly beats MLP-based counterparts on multi-scale Poisson (high+low frequency), crack-tip singularity, plate-with-hole stress concentration, large-deformation hyperelasticity and material-heterogeneous problems, with up to an order-of-magnitude smaller L2 error using fewer parameters. MLPs remain superior on complex boundary geometries where KAN's spline grid does not naturally adapt.
