---
slot: 87
title: "FO-PINN: A First-Order formulation for Physics-Informed Neural Networks"
authors: [Rini Jasmine Gladstone, Mohammad Amin Nabian, N. Sukumar, Ankit Srivastava, Hadi Meidani]
year: 2025
venue: "Engineering Analysis with Boundary Elements 174 (2025) 106161"
gitrepo: ""
doi: "10.1016/j.enganabound.2025.106161"
---

## TL;DR
For a `d`-th order PDE, FO-PINNs make the network output not only the field `u` but also all derivatives of order up to `d-1`, and recast the PDE as a system of **first-order** equations with explicit *compatibility* losses. Only first-order autograd is needed - cutting training time ~2-3x - and exact hard boundary conditions via R-function ADFs become tractable because the Laplacian is no longer "exploding" at corners.

## Problem
Second/higher-order autograd in standard PINNs is expensive and exhibits sharp variations that destabilise training, harming parametric problems. Approximate Distance Function (ADF) hard-BC ansatzes are excellent for first-order PDEs but fail at non-smooth boundary joints when applied to second-order operators (Laplacian blows up at corners). Automatic Mixed Precision (AMP) cannot be used because second-order gradients need different scaling.

## Method
For a PDE `a u_xx + b u_xy + c u_yy + d u_x + e u_y + f u = g` on `Omega`, introduce auxiliary outputs `u_x_hat, u_y_hat` predicted by the network. The reformulated PDE and compatibility losses are:
$$ a\,\partial_x \hat u_x + b\,\partial_x \hat u_y + c\,\partial_y \hat u_y + d\,\hat u_x + e\,\hat u_y + f\,u = g $$
$$ \mathcal{J}_{\text{comp}} = \frac{1}{m}\sum_j \Big( \hat u_x - \partial_x u \Big)^2_j + \Big( \hat u_y - \partial_y u \Big)^2_j $$
Total loss `J_FOPINN = J_PDE + lambda_C J_comp`. BCs are imposed *exactly* via the ADF ansatz `u_sol = g(x) + phi(x) u_net(x)` where `phi` is built from R-function disjunction over each boundary piece `phi_i`. For Neumann/Robin similar but with directional derivative.

R-function ADF for `n` boundary pieces normalised to order `m`:
$$ \phi(\mathbf{x}) = \Big( \phi_1^{-m} + \phi_2^{-m} + \dots + \phi_n^{-m} \Big)^{-1/m} $$

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class FOPINN(nn.Module):
    H: int = 256
    depth: int = 4
    d_out: int = 1
    n_aux: int = 2                                  # u_x, u_y
    @nn.compact
    def __call__(self, x):
        h = x
        for _ in range(self.depth):
            h = nn.silu(nn.Dense(self.H)(h))
        out = nn.Dense(self.d_out + self.n_aux)(h)
        return out[..., :self.d_out], out[..., self.d_out:]

def line_adf(x, p1, p2):                            # eq. (6)
    L = jnp.linalg.norm(p2 - p1)
    t = jnp.sum((x - p1) * (p2 - p1), axis=-1) / L
    f = (L**2 - jnp.sum((x - (p1 + p2) / 2)**2, axis=-1)) / L
    Phi = jnp.sqrt(t**2 + f**4 + 1e-12)
    return jnp.sqrt(f**2 + 0.25 * (Phi - t)**2 + 1e-12)

def adf_R(phis, m=2):                               # R-equivalence
    inv = sum(phi**(-m) for phi in phis)
    return inv ** (-1.0 / m)

def helmholtz_fopinn_loss(params, x, k, g_bc_func, vertices, lam_C=1.0):
    def u_sol_at(xi):
        u_net, aux = FOPINN().apply(params, xi)
        phis = [line_adf(xi, vertices[i], vertices[(i+1) % len(vertices)])
                for i in range(len(vertices))]
        phi  = adf_R(phis, m=2)
        return (g_bc_func(xi) + phi * u_net).squeeze(), aux

    def point_residuals(xi):
        u_sol, aux = u_sol_at(xi)
        ux_pred, uy_pred = aux[0], aux[1]
        # First-order grads of predicted aux fields
        ux_pred_fn = lambda y: FOPINN().apply(params, y)[1][0]
        uy_pred_fn = lambda y: FOPINN().apply(params, y)[1][1]
        dux = jax.grad(ux_pred_fn)(xi)              # (2,)
        duy = jax.grad(uy_pred_fn)(xi)
        # gradient of u_sol (first-order only)
        du = jax.grad(lambda y: u_sol_at(y)[0])(xi)
        pde_res = k**2 * u_sol + dux[0] + duy[1]
        comp = (ux_pred - du[0])**2 + (uy_pred - du[1])**2
        return pde_res**2, comp

    pde_sq, comp_sq = jax.vmap(point_residuals)(x)
    return jnp.mean(pde_sq) + lam_C * jnp.mean(comp_sq)

optimizer = optax.adam(1e-3); opt_state = optimizer.init(params)
```

Hyperparameters: 1-4 hidden layers x 256 with Swish; Adam lr `1e-3` with NTK-style learning-rate annealing for `(lam_C, lam_B)`; 1000 interior + 100 boundary points per batch; 30k epochs; non-dimensionalise to unit-length domain. JAX runs in float32 by default; only first-order gradients - no second-order operator to break mixed-precision pipelines.

## Results
- Helmholtz square: standard PINN soft-BC `1.1e-2` rel-L2; FO-PINN exact-BC `4.2e-3` (~2.6x better, with `u = 0` exactly on boundary).
- Annular-ring Navier-Stokes: ~2.2x faster per iteration; AMP adds another 1.33x (~2.9x total).
- Parametric Navier-Stokes and 4th-order PDE: standard PINN accuracy collapses with parameter count; FO-PINN stays accurate.
