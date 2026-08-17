---
slot: 055
title: "Discontinuity Computing Using Physics-Informed Neural Networks"
authors: [Li Liu, Shengping Liu, Hui Xie, Fansheng Xiong, Heng Yong]
year: 2022
venue: Journal of Scientific Computing (arXiv:2206.03864)
gitrepo: ""
---

## TL;DR
Plain PINNs stall on conservation laws with shocks because residual points falling inside the shock carry infinite-gradient loss they cannot reduce. PINN-WE multiplies the PDE residual by a local **compression-sensitive weight** `lambda(x)` that shrinks near shocks, lets the network fit the smooth regions, and lets physical compression squeeze the transition points into a sharp discontinuity automatically.

## Problem
For hyperbolic conservation laws `U_t + div F(U) = 0`, the strong-form residual is undefined at shocks (infinite gradient). Transition points that fall inside a shock dominate `L_PDE`, but neither raising nor lowering the gradient there can reduce the loss — training is stuck in a paradox and the smooth regions also fail to converge. Vanilla PINN therefore cannot capture Sod, Lax, or 2D Riemann shocks accurately.

## Method
Replace the residual `G = U_t + div F(U)` by a **weighted residual** `G_new = lambda(x) * G` with a strictly positive, gradient-dependent scalar weight that detects compression (negative velocity divergence):

$$
\lambda(x) = \frac{1}{\varepsilon_2 \,(|\nabla\!\cdot\!\vec u| - \nabla\!\cdot\!\vec u) + 1}
$$

So `lambda = 1` in smooth or expansive regions (`div u >= 0`) and `lambda -> 0` inside strong compression (`div u << 0`). The PDE loss becomes the mean of `G_new^2` over collocation points, plus the standard IC/BC mismatch:

$$
\mathcal{L} = \frac{1}{|S_{PDE}|}\sum_i \big(\lambda(x_i)\,G(x_i)\big)^2 + \varepsilon_1\,\mathcal{L}_{IBs}
$$

Because `lambda > 0`, weighted-zero residual has the same classical solutions; it only reweights training. For scalar Burgers, `div u` becomes `du/dx`. No artificial viscosity is added to the equations themselves — only to the loss weight. Recommended scalar: `eps_2 ~ 1` (problem-dependent), `eps_1` 10-100 (BC weight).

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    width: int = 50
    depth: int = 7
    out_dim: int = 3
    @nn.compact
    def __call__(self, xt):
        h = xt
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width)(h))
        return nn.Dense(self.out_dim)(h)

def euler_residual_1d(params, apply_fn, x, t):
    def U_of(xs, ts):
        xt = jnp.stack([xs, ts])
        return apply_fn(params, xt)            # [3]
    def rho_u_E(xs, ts):
        U = U_of(xs, ts)
        rho, mom, E = U[0], U[1], U[2]
        u = mom / rho
        return rho, mom, E, u
    def prim(xs, ts):
        rho, mom, E, u = rho_u_E(xs, ts)
        p  = (1.4 - 1.0) * (E - 0.5 * rho * u**2)
        return rho, mom, E, u, p
    def flux_t_terms(xs, ts):
        rho, mom, E, u, p = prim(xs, ts)
        F1 = mom
        F2 = mom * u + p
        F3 = u * (E + p)
        return jnp.array([rho, mom, E]), jnp.array([F1, F2, F3]), u
    drho_dt = jax.grad(lambda ts: rho_u_E(x, ts)[0])(t)
    dmom_dt = jax.grad(lambda ts: rho_u_E(x, ts)[1])(t)
    dE_dt   = jax.grad(lambda ts: rho_u_E(x, ts)[2])(t)
    dF1_dx  = jax.grad(lambda xs: flux_t_terms(xs, t)[1][0])(x)
    dF2_dx  = jax.grad(lambda xs: flux_t_terms(xs, t)[1][1])(x)
    dF3_dx  = jax.grad(lambda xs: flux_t_terms(xs, t)[1][2])(x)
    du_dx   = jax.grad(lambda xs: rho_u_E(xs, t)[3])(x)
    G1, G2, G3 = drho_dt + dF1_dx, dmom_dt + dF2_dx, dE_dt + dF3_dx
    eps2 = 1.0
    lam = 1.0 / (eps2 * (jnp.abs(du_dx) - du_dx) + 1.0)   # >0, <=1
    return lam * G1, lam * G2, lam * G3

batched_res = jax.vmap(euler_residual_1d, in_axes=(None, None, 0, 0))

@jax.jit
def train_step(params, opt_state, x_r, t_r, xt_ib, U_ib, apply_fn, optimizer):
    def loss(p):
        g1, g2, g3 = batched_res(p, apply_fn, x_r, t_r)
        L_pde = jnp.mean(g1**2 + g2**2 + g3**2)
        L_ib  = jnp.mean((jax.vmap(apply_fn, in_axes=(None, 0))(p, xt_ib) - U_ib)**2)
        return L_pde + 10.0 * L_ib
    grads = jax.grad(loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
# optional: switch to L-BFGS (jaxopt) for final refinement
```

Network: 4-7 hidden layers, 30-50 tanh neurons, 10k uniform-grid residual points, ~1k IB points; Adam(1e-3) then L-BFGS.

## Results
On inviscid Burgers, Sod, Lax, 2D Riemann, and an oblique moving shock, PINN-WE produces sharper, oscillation-free shocks than fifth-order WENO-Z on comparable meshes; vanilla PINN plateaus around `L_total ~ 1e-1` while PINN-WE drives both `L_PDE` and `L_IBs` 1-2 orders of magnitude lower. The transition points get compressed into a near-zero-thickness front automatically.
