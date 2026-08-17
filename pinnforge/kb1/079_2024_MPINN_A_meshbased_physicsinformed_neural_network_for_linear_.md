---
slot: 79
title: "M-PINN: A mesh-based physics-informed neural network for linear elastic problems in solid mechanics"
authors: [Lu Wang, Guangyan Liu, Guanglun Wang, Kai Zhang]
year: 2024
venue: "International Journal for Numerical Methods in Engineering 125 (2024)"
gitrepo: ""
doi: "10.1002/nme.7444"
---

## TL;DR
M-PINN augments the standard PINN loss with a **FEM-reconstruction** term: the predicted displacement field is projected onto isoparametric finite elements, and the network is required to match this reconstruction. This injects an FEM-style prior data distribution and lets the method work even when boundary conditions are unknown.

## Problem
For 2-D linear elasticity (open-hole plate under tension), vanilla PINN struggles with stress concentration and is brittle when full boundary data is missing - the residual-only loss has too large a search space.

## Method
Two losses are summed: `L = L_r + L_f`. `L_r` is the standard residual (Navier equation + any known BCs). `L_f` is the FEM-reconstruction loss:

1. **Domain meshing**: partition Omega into `M` coarse 8-node quadrilateral isoparametric elements.
2. **Inner FEM optimisation**: for each element with nodal displacements `u^{m,i}` (unknown), define `h(x,y) = sum_i N_i(xi,eta) u^{m,i}` where `N_i` are FEM shape functions. Minimise
$$ \mathcal{M}_m = \frac{1}{N_m}\sum_{(x,y)\in\Omega_m} \big( g(x,y) - h(x,y;\,u^{m,i})\big)^2 $$
where `g(x,y)` is the concatenation of network prediction `u_hat` and any observed `u_obv`. Solving yields `u_hat_f` - the closest FEM-admissible field.
3. **Outer PINN training**: minimise `L_f = (1/M) sum_m |u_hat - u_hat_f|^2` against the network. `u_hat_f` is treated as a target (detached).

PDE residual is the Navier-Cauchy equation:
$$ \frac{E}{1-\nu^2}\Big(\frac{1-\nu}{2}\nabla^2 u_i + \frac{1+\nu}{2}\partial_i(\nabla\cdot \mathbf{u})\Big) + f_i = 0 $$

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MPINN(nn.Module):
    H: int = 64
    depth: int = 5
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.H)(x))
        return nn.Dense(2)(x)                       # (u, v)

def navier_residual(params, x, E, nu, f=0.0):
    u_of = lambda xi: MPINN().apply(params, xi)     # (2,)
    # Hessian per component:
    def res_point(xi):
        H_ux = jax.hessian(lambda y: u_of(y)[0])(xi)
        H_uy = jax.hessian(lambda y: u_of(y)[1])(xi)
        lap_ux = jnp.trace(H_ux); lap_uy = jnp.trace(H_uy)
        div_x  = H_ux[0, 0] + H_uy[0, 1]            # d/dx of div u
        div_y  = H_ux[1, 0] + H_uy[1, 1]            # d/dy of div u
        coef = E / (1 - nu**2)
        rx = coef * ((1-nu)/2 * lap_ux + (1+nu)/2 * div_x) + f
        ry = coef * ((1-nu)/2 * lap_uy + (1+nu)/2 * div_y) + f
        return jnp.array([rx, ry])
    return jax.vmap(res_point)(x)

def fem_reconstruct(u_pred_per_elem, gauss_xy_per_elem):
    """Inner LS solve: find nodal disp u^{m,i} that best fit u_pred on each elem."""
    N = shape_8node(gauss_xy_per_elem)              # (Nq, 8)
    U = jnp.linalg.lstsq(N, u_pred_per_elem)[0]     # (8, 2)
    return N @ U                                    # reconstructed field

def loss_fn(params, x_col, x_bc, u_bc, elem_idx, gauss_xy, E, nu, lam_f):
    u_pred = MPINN().apply(params, x_col)
    r      = navier_residual(params, x_col, E, nu)
    L_r    = jnp.mean(r**2)
    # FEM reconstruction (stop-gradient on target)
    u_recon = jnp.concatenate([fem_reconstruct(u_pred[elem_idx[m]], gauss_xy[m])
                                for m in range(len(elem_idx))])
    L_f = jnp.mean((u_pred - jax.lax.stop_gradient(u_recon))**2)
    return L_r + lam_f * L_f

optimizer = optax.adam(1e-3); opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, *args):
    grads = jax.grad(loss_fn)(params, *args)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state
```

Hyperparameters: tanh MLP 5 x 64; 8-node Serendipity isoparametric elements (`M ~ 8-32` elements for an open-hole plate); collocation density >> mesh density; `lambda_f` in `[1, 100]`; inner FEM step re-run every k outer iters (e.g. k=10) to amortise cost.

## Results
On the open-hole tension benchmark (Young's modulus 10 GPa, `nu=0.3`, prescribed top displacement 0.5 m), M-PINN reduces displacement L2 error vs vanilla PINN, and is the *only* method to produce sensible fields when one boundary condition is removed.
