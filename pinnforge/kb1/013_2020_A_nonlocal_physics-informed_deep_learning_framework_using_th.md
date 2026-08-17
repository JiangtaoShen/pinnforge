---
slot: 13
title: "A nonlocal physics-informed deep learning framework using the peridynamic differential operator"
authors: [E. Haghighat, A. Bekar, E. Madenci, R. Juanes]
year: 2020
venue: "arXiv:2006.00446"
gitrepo: ""
---

## TL;DR
Replace pointwise autograd derivatives in PINN by the *Peridynamic Differential Operator* (PDDO): derivatives `df/dx, d2f/dx2, ...` are evaluated at point `x` as weighted spatial *integrals* over a family `H_x` of nearby points within a *horizon* `delta`. This injects nonlocal/long-range interactions into the inputs and gives derivatives that are accurate near sharp gradients and boundaries, where autograd-PINN degrades.

## Problem
For mixed displacement-traction BCs in solid mechanics (rigid punch indentation), the solution develops localised steep stress gradients / singularities. Local PINN (autograd derivatives + MLP) cannot represent these globally and the loss stalls.

## Method
For each "centre" point `x` precompute its family `H_x = {x_j : |x - x_j| <= delta}` and exponential weight
$$
w(|\xi|) = \exp(-4|\xi|^2/\delta^2),\qquad \xi_j = x_j - x
$$
Madenci's PDDO produces a set of *kernel functions* `g_{p1 p2}(xi)` (closed-form, one per derivative order) such that
$$
f(x) \approx \sum_{j\in H_x} f(x_j)\,g_{00}(\xi_j)\,A_j,\qquad
\frac{\partial f}{\partial x}\big|_x \approx \sum_{j\in H_x} f(x_j)\,g_{10}(\xi_j)\,A_j
$$
and similarly for `g_{01}, g_{20}, g_{02}, g_{11}` for second-order derivatives. The `g_{p1 p2}` are obtained once by enforcing Taylor-series orthogonality; they are tabulated in the paper.

The PINN ansatz becomes a *set-valued* network: given centre `x` and its precomputed family `(x_1,...,x_N)`, the network outputs `f(x), f(x_1), ..., f(x_N)`. From these, derivatives at `x` are computed by the PDDO quadrature above (matrix-vector product with fixed `g`-weights).

Two variants:
- **PDDO-PINN**: use PDDO for *all* spatial derivatives in the loss (also no autograd).
- **AD-PDDO-PINN**: PDDO only for spatial discretisation of nonlocal kernel; AD still used elsewhere.

Loss (linear elasticity example): equilibrium + constitutive + data, computed at sampling centres using PDDO derivatives:
$$
\mathcal{L} = \sum_i |\sigma_{ij,j}^{PDDO}(x_i)|^2 + \text{constitutive residuals} + \text{data terms}
$$

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class PDDO_PINN(nn.Module):
    hidden: int = 40
    depth: int = 5
    out_dim: int = 1
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(self.out_dim)(x)

# Pre-compute family and PDDO weights once (NumPy / JAX on host, treated as constants).
def build_pddo(X, delta, order=2):
    # X: (M, d) centres == samples. Returns padded neighbour indices + g-weights.
    M = X.shape[0]
    fam_idx = []
    g00, g10, g01, g20, g02, g11 = [], [], [], [], [], []
    A_vol_all = compute_voronoi_areas(X)
    for i in range(M):
        dxi = X - X[i]
        r2  = (dxi ** 2).sum(-1)
        mask = (r2 < delta ** 2) & (r2 > 0)
        idx  = jnp.where(mask, size=64, fill_value=i)[0]    # fixed-size pad
        xij  = X[idx] - X[i]
        w    = jnp.exp(-4 * (xij ** 2).sum(-1) / delta ** 2)
        A_j  = A_vol_all[idx]
        fam_idx.append(idx)
        g00.append(pddo_kernels(xij, w, A_j, order=(0, 0)))
        g10.append(pddo_kernels(xij, w, A_j, order=(1, 0)))
        # ... g01, g20, g02, g11 likewise
    return jnp.stack(fam_idx), (jnp.stack(g00), jnp.stack(g10), ...)

fam_idx, G = build_pddo(X_centres, delta=0.1, order=2)

net = PDDO_PINN(out_dim=2)
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

def derivs(f_all, fam, G):                            # f_all: (M, ...) → derivatives at every centre
    f_j = f_all[fam]                                   # (M, K, ...)
    df_dx  = (f_j * G[1][:, :, None]).sum(1)
    df_dy  = (f_j * G[2][:, :, None]).sum(1)
    d2f_xx = (f_j * G[3][:, :, None]).sum(1)
    d2f_yy = (f_j * G[4][:, :, None]).sum(1)
    d2f_xy = (f_j * G[5][:, :, None]).sum(1)
    return df_dx, df_dy, d2f_xx, d2f_yy, d2f_xy

def loss_fn(params, X_centres):
    u_all = net.apply(params, X_centres)               # (M, 2)
    ux_x, ux_y, *_ = derivs(u_all[:, 0:1], fam_idx, G)
    uy_x, uy_y, *_ = derivs(u_all[:, 1:2], fam_idx, G)
    return jnp.mean(equilibrium_residual(ux_x, ux_y, uy_x, uy_y) ** 2)

@jax.jit
def train_step(params, opt_state, X_centres):
    grads = jax.grad(loss_fn)(params, X_centres)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

Recommended: horizon `delta = 3 dx` where `dx` is mean inter-point spacing; PDDO order 2; depth 5, width 40, tanh; Adam `lr=1e-3` then L-BFGS.

## Results
On rigid-punch indentation of an elastoplastic strip (mixed displacement-traction BC, stress concentration at punch corners), local PINN errs by 20-50% in stress near the singularity while PDDO-PINN matches FEM reference within 2-5%. Inverse identification of `mu` (shear modulus) and `sigma_Y0` (yield stress) from sparse interior data: PDDO-PINN <1% error vs 10-20% for local PINN.
