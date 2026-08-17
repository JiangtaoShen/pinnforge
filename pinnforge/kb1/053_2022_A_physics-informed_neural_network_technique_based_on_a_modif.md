---
slot: 053
title: "A physics-informed neural network technique based on a modified loss function for computational 2D and 3D solid mechanics"
authors: [Jinshuai Bai, Timon Rabczuk, Ashish Gupta, Laith Alzubaidi, Yuantong Gu]
year: 2022
venue: Computational Mechanics 71:543-562 (2023)
gitrepo: "https://github.com/JinshuaiBai/LSWR_loss_function_PINN"
doi: 10.1007/s00466-022-02252-0
---

## TL;DR
Replace the pointwise MSE PDE loss with an *integrated* Least-Squares Weighted Residual (LSWR) loss `∫_Ω R_g² dΩ + ∫_Γt R_t² dΓ`, scaled by `χ_1 = h/E²` and `χ_2 = 1/E²` to be dimensionless. Numerical quadrature uses Delaunay triangulation for arbitrary point distributions. The result is a PINN that, with a single tunable parameter `h`, predicts both displacement and stress fields better than the collocation-MSE and the DEM energy-based losses on 2-D/3-D linear and geometrically nonlinear solid mechanics.

## Problem
The collocation MSE loss `Σ R²(x_i)` only enforces the residual at sample points and generalises poorly between them. The DEM energy loss `½∫σ:ε dV − ∫u·t dA` smooths stresses and does not strictly satisfy the equilibrium equation. Both also have unit-imbalanced terms with no clean dimensionless form.

## Method
**LSWR loss (dimensionless).**
$$ \mathcal{L} = \chi_1 \int_\Omega R_g^2\,d\Omega + \chi_2 \int_{\Gamma_t} R_t^2\,d\Gamma_t,\qquad \chi_1 = \tfrac{h}{E^2},\quad \chi_2 = \tfrac{1}{E^2} $$
with residuals
$$ R_g = \sigma_{\alpha\beta,\beta} + f_\alpha,\qquad R_t = \sigma_{\alpha\beta} n_\beta - \bar t_\alpha $$
Dirichlet BCs are imposed hard via the boundary-imposition ansatz (`u(x) = A(x) + B(x)·NN(x)`). `h` is the only hyperparameter — close to the average inter-point spacing.

**Delaunay quadrature for arbitrary point clouds.** Triangulate (2D) or tetrahedralise (3D) the sample points; each triangle's centroid-to-side-midpoint subdivision assigns one-third of each adjacent triangle's area to its vertex. Discrete loss:
$$ \mathcal{L} \approx \chi_1 \sum_{i=1}^{n} s_i R_g^2(x_i) + \chi_2 \sum_{i=1}^{m_t} \ell_i R_t^2(x_i) $$
For regularly spaced points, second-order Gauss quadrature is used instead.

**Network.** Single FFN with output `(u_x, u_y[, u_z])`; stresses derived from `u` via constitutive law and autograd (Cauchy/PK1/PK2 depending on linear vs. geometrically nonlinear case).

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from math import sqrt
from scipy.spatial import Delaunay

class PINN(nn.Module):
    d_out:  int = 2
    hidden: int = 40
    depth:  int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(self.d_out)(x)

def linear_elastic_residuals(params, X, lam, mu):
    def u_of(z): return model.apply(params, z)
    # ∇u: per-point jacobian (N, 2, 2)
    du = jax.vmap(jax.jacrev(lambda x: u_of(x[None])[0]))(X)
    du_x, du_y = du[:, 0, :], du[:, 1, :]
    eps_xx = du_x[:, 0:1]; eps_yy = du_y[:, 1:2]
    eps_xy = 0.5 * (du_x[:, 1:2] + du_y[:, 0:1])
    s_xx = (lam + 2 * mu) * eps_xx + lam * eps_yy
    s_yy = lam * eps_xx + (lam + 2 * mu) * eps_yy
    s_xy = 2 * mu * eps_xy
    # divergence of σ (per component, first-order autograd)
    ds_xx = jax.vmap(jax.grad(lambda z:
        ((lam + 2 * mu) * jax.grad(lambda zz: u_of(zz[None])[0, 0])(z)[0]
          + lam *           jax.grad(lambda zz: u_of(zz[None])[0, 1])(z)[1])))(X)
    # … similar for s_xy, s_yy …  (omitted for brevity; see paper for full assembly)
    Rg_x = ds_xx[:, 0:1] + grad_y_of_sxy(params, X)
    Rg_y = grad_x_of_sxy(params, X) + grad_y_of_syy(params, X)
    return Rg_x, Rg_y, (s_xx, s_yy, s_xy)

def delaunay_areas(pts):
    tri = Delaunay(np.asarray(pts)).simplices
    s = np.zeros(len(pts))
    for t in tri:
        v = np.asarray(pts[t])
        a = 0.5 * abs((v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1])
                     -(v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1]))
        s[t] += a / 3.0
    return jnp.asarray(s)

E, nu  = 1.0e3, 0.3
mu     = E / (2 * (1 + nu))
lam    = E * nu / ((1 + nu) * (1 - 2 * nu))
h      = 1.0 / sqrt(n_points)
chi1, chi2 = h / E ** 2, 1.0 / E ** 2

model  = PINN()
params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
areas  = jax.lax.stop_gradient(delaunay_areas(pts))

def loss_fn(params, pts, lengths_t, n_b, t_b):
    Rg_x, Rg_y, (sxx, syy, sxy) = linear_elastic_residuals(params, pts, lam, mu)
    L_int = chi1 * jnp.sum(areas * (Rg_x ** 2 + Rg_y ** 2).squeeze())
    Rt    = traction_residual(sxx, syy, sxy, n_b, t_b)
    L_bnd = chi2 * jnp.sum(lengths_t * Rt ** 2)
    return L_int + L_bnd

opt = optax.adam(1e-3)
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, pts, lengths_t, n_b, t_b):
    g = jax.grad(loss_fn)(params, pts, lengths_t, n_b, t_b)
    upd, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state
```

Recommended hyperparameters: tanh MLP 4 layers × 40; Adam lr=1e-3 → L-BFGS; sample density chosen so `h` ≈ average spacing; for nonlinear problems use Green-Lagrange strain, PK1/PK2 stress, transfer-learning from the linear solution.

## Results
Tested on pure bending of a 2-D beam, plate with hole, Cook's membrane, and 3-D linear/large-deformation problems. The LSWR PINN matches FEM displacement and stress fields across the domain, while collocation-MSE PINN over-fits to the sample points (poor generalisation between them) and DEM under-predicts stress peaks. The single parameter `h` is robust over an order of magnitude and is essentially the inter-point spacing.
