---
slot: 049
title: "The mixed deep energy method for resolving concentration features in finite strain hyperelasticity"
authors: [J. Fuhg, N. Bouklas]
year: 2021
venue: Journal of Computational Physics (arXiv:2104.09623)
gitrepo: ""
---

## TL;DR
mDEM augments the Deep Energy Method (DEM) — which minimises the elastic potential energy — with additional NN outputs for the first Piola-Kirchhoff stress components and a constitutive consistency loss `P̂ = P(F̂)`. This recovers sharp displacement/stress concentrations that pure-displacement PINN and DEM smear out, while needing only first-order autograd. A Delaunay-triangulation quadrature lets the energy be integrated accurately on irregular point clouds around holes/notches.

## Problem
Both PINN (strong form, requires 2nd-order autograd through `div P`) and DEM (energy form, 1st-order autograd) under-resolve stress concentrations in solid mechanics — the global shape function over-smooths near holes, notches, and concentrated loads — and Neumann tractions are only weakly enforced.

## Method
**Hyperelastic background.** Deformation gradient `F = I + ∇u`, Neo-Hookean strain energy:
$$ \Psi = \tfrac{1}{4}\lambda(\log J^2 - 1 - 2\log J) + \tfrac{1}{2}\mu(\mathrm{tr}\,C - 2 - 2\log J),\quad J=\det F,\ C=F^T F $$
First Piola: `P = ∂Ψ/∂F`. Total potential:
$$ \Pi(\varphi) = \int_B \Psi\,dV - \int_B f_b\!\cdot\!\varphi\,dV - \int_{\Gamma_t} \tilde t\!\cdot\!\varphi\,dA $$

**mDEM architecture.** Single NN, input `X`, outputs `(z, Z) ∈ R^{2+4}` in 2D: `û = A(X)+B(X)∘z` (Dirichlet hard) and `P̂ = C(X)+D(X)∘Z` (Neumann hard if available). Loss:
$$ \mathcal{L}(\Theta) = \Pi(\hat\varphi) + W_P\,\mathrm{MSE}_P + W_u\,\mathrm{MSE}_u + W_t\,\mathrm{MSE}_t $$
$$ \mathrm{MSE}_P = \tfrac{1}{N_\Pi}\sum_i \|\hat P(X_i) - P(\hat F(X_i))\|^2,\quad \mathrm{MSE}_t = \tfrac{1}{N_t}\sum_i \|\hat P(X_i)\,N_i - \tilde t\|^2 $$
The constitutive loss `MSE_P` ties the auxiliary stress outputs to those derived from `û`, regularising both fields.

**Delaunay integration of `Π`.** Triangulate the collocation points, then on each triangle apply standard Gauss quadrature; total energy = sum over triangles. This handles random/irregular point distributions around geometric features (constrained Voronoi/Delaunay near holes).

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from scipy.spatial import Delaunay

class mDEM(nn.Module):
    hidden: int = 40
    depth:  int = 4

    @nn.compact
    def __call__(self, X):
        h = X
        for _ in range(self.depth):
            h = jnp.tanh(nn.Dense(self.hidden)(h))
        u = nn.Dense(2, name="head_u")(h)               # displacements
        P = nn.Dense(4, name="head_P")(h).reshape(-1, 2, 2)
        return u, P

def deformation_gradient(u_fn, params, X):
    # ∂u_i/∂X_j  → (B, 2, 2)
    G = jax.vmap(jax.jacrev(lambda x: u_fn(params, x[None])[0][0]))(X)
    return jnp.eye(2)[None] + G

def neohookean_P(F, lam, mu):
    J      = jnp.linalg.det(F)
    Finv_T = jnp.linalg.inv(F).swapaxes(-1, -2)
    # P = μ(F - F^{-T}) + λ log(J) F^{-T}
    return mu * (F - Finv_T) + lam * jnp.log(J)[..., None, None] * Finv_T

def delaunay_quadrature(params, X_pts, lam, mu, tri):
    def model_apply(p, x): return model.apply(p, x)
    u_hat, P_hat = model_apply(params, X_pts)
    F = deformation_gradient(lambda p, x: model_apply(p, x), params, X_pts)
    J = jnp.linalg.det(F)
    C = F.swapaxes(-1, -2) @ F
    Psi = (0.25 * lam * (jnp.log(J ** 2) - 1 - 2 * jnp.log(J))
           + 0.5 * mu * (jnp.einsum('bii->b', C) - 2 - 2 * jnp.log(J)))
    # accumulate Π over triangles (linear element, 1-point quadrature)
    v0, v1, v2 = X_pts[tri[:, 0]], X_pts[tri[:, 1]], X_pts[tri[:, 2]]
    area = 0.5 * jnp.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                         -(v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
    Psi_mean = (Psi[tri[:, 0]] + Psi[tri[:, 1]] + Psi[tri[:, 2]]) / 3.0
    Pi = jnp.sum(area * Psi_mean)
    # constitutive loss
    P_from_u = neohookean_P(F, lam, mu)
    L_P = jnp.mean((P_hat - P_from_u) ** 2)
    return Pi, L_P, P_hat, u_hat

model     = mDEM()
params    = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
tri       = jnp.array(Delaunay(np.asarray(X_pts)).simplices)
opt       = optax.adam(1e-3)
opt_state = opt.init(params)

def total_loss(params, X_pts, idx_t, normals, t_pres, idx_u, u_pres, lam, mu, tri):
    Pi, L_P, P_hat, u_hat = delaunay_quadrature(params, X_pts, lam, mu, tri)
    L_t = traction_loss(P_hat[idx_t], normals[idx_t], t_pres)
    L_u = jnp.mean((u_hat[idx_u] - u_pres) ** 2)
    return Pi + W_P * L_P + W_t * L_t + W_u * L_u

@jax.jit
def step(params, opt_state, X_pts, idx_t, normals, t_pres, idx_u, u_pres):
    g = jax.grad(total_loss)(params, X_pts, idx_t, normals, t_pres,
                             idx_u, u_pres, lam, mu, tri)
    upd, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state
```

Recommended: tanh MLP 4 layers × 40, single network with two heads (`u`, `P`); weights `W_P, W_t, W_u` tuned by gradient-balancing or set ~1; Adam → L-BFGS; Delaunay triangulation pre-computed on the reference configuration; impose `u`-Dirichlet hard via `A, B`.

## Results
On plane-strain hyperelastic plates with holes and notches subject to concentrated loads, mDEM matches FEM stress concentration factors closely, whereas pure DEM and displacement-PINN under-predict peaks by 15-40%. Delaunay quadrature is essential for accurate `Π` on irregular point sets; the mixed (u, P) output drastically improves Neumann boundary satisfaction.
