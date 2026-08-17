---
slot: 031
title: "A two-stage physics-informed neural network method based on conserved quantities and applications in localized wave solutions"
authors: [Shuning Lin, Yong Chen]
year: 2021
venue: "J. Comp. Phys. (arXiv:2107.01009)"
gitrepo: ""
---

## TL;DR
For integrable PDEs (Boussinesq-Burgers, Sawada-Kotera, classical Boussinesq-Burgers), augment standard PINN with a second training stage that adds a GLOBAL loss term penalizing departure from a known conserved quantity m(t) = integral of M[u]dx. The conservation MSE imposes a non-local constraint that vanilla pointwise PINN can't capture and recovers solitons, M-shape and plateau soliton solutions accurately.

## Problem
Vanilla PINN loss is only local (residual + initial/boundary MSE at points). For integrable systems with rich soliton structure (soliton molecules, interaction solutions), local constraints leave global drift — mass, momentum, energy are not preserved and the network produces wrong wave amplitudes/positions far from the IC.

## Method
**Stage 1:** Train standard PINN to minimize
$$
\mathrm{MSE}_1 = \mathrm{MSE}_u + \mathrm{MSE}_f
$$
Call the resulting prediction u_1.

**Stage 2:** Re-initialize a fresh net (or keep u_1's weights) and minimize
$$
\mathrm{MSE}_2 = \mathrm{MSE}_u + \mathrm{MSE}_f + \mathrm{MSE}_s + \mathrm{MSE}_m
$$
with
- MSE_s = (1/N_s) sum |hat_u(x_s, t_s) - u_1(x_s, t_s)|^2: trust-region toward stage-one solution.
- MSE_m = (1/N_c) sum_{i=1..N_c} |m(t_i) - m(t_0)|^2: conserved-quantity constraint.

For conserved density M[u] (e.g. M=u for mass, M=u^2 for L2-norm, M= some H from Lax pair):
$$
m(t) = \int_{x_0}^{x_1} M[u](x,t)\,dx \approx \frac{x_1-x_0}{N_x-1}\sum_{j=2}^{N_x} M[\hat u](x_j, t)
$$
m(t_0) is computed once from the analytic/initial data and treated as the target.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax, jaxopt

def integrate_M(params, apply_fn, M_op, x_grid, t_query):
    xt   = jnp.stack([x_grid, jnp.full_like(x_grid, t_query)], axis=-1)
    Mu   = jax.vmap(lambda z: M_op(params, apply_fn, z))(xt)
    dx   = (x_grid[-1] - x_grid[0]) / (x_grid.shape[0] - 1)
    return jnp.sum(Mu) * dx

def stage_two_loss(params, params1, apply_fn,
                   x_u, u_data, x_f, x_s, t_query_list,
                   M_op, x_grid, m_t0):
    L_u = jnp.mean((jax.vmap(lambda z: apply_fn(params, z))(x_u) - u_data)**2)
    L_f = jnp.mean(jax.vmap(lambda z: pde_residual(params, apply_fn, z))(x_f)**2)
    # trust region toward stage-1 solution (frozen)
    u1  = jax.lax.stop_gradient(jax.vmap(lambda z: apply_fn(params1, z))(x_s))
    L_s = jnp.mean((jax.vmap(lambda z: apply_fn(params, z))(x_s) - u1)**2)
    # conservation
    mts = jax.vmap(lambda t: integrate_M(params, apply_fn, M_op, x_grid, t))(t_query_list)
    L_m = jnp.mean((mts - m_t0)**2)
    return L_u + L_f + L_s + L_m

# Stage 1
params1 = train_pinn_basic(net, key)        # standard PINN
# Stage 2: warm-start from params1, optimize with L-BFGS (jaxopt)
params  = jax.tree_util.tree_map(lambda x: x, params1)
solver = jaxopt.LBFGS(
    fun=lambda p: stage_two_loss(p, params1, net.apply,
                                 x_u, u_data, x_f, x_s, t_query_list,
                                 M_op, x_grid, m_t0),
    linesearch="zoom",
    maxiter=20000)
params, _ = solver.run(params)
```

Recommended: tanh MLP, L-BFGS (since loss is now smooth and stage-1 provides good init), N_c ~ 20-50 conservation checkpoints, choose M[u] = first non-trivial conserved density of the system (consult Lax pair).

## Results
On Boussinesq-Burgers one-soliton: relative L2 from 1.8e-3 (stage 1) to 4e-4 (stage 2). On classical BB interaction solutions and Sawada-Kotera soliton-molecule / M-shape / plateau solutions, error reductions of 30-80% over single-stage PINN, with better long-time wave-shape preservation.
