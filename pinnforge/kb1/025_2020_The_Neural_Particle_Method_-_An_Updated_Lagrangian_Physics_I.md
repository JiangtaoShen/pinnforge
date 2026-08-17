---
slot: 025
title: "The Neural Particle Method - An Updated Lagrangian Physics Informed Neural Network for Computational Fluid Dynamics"
authors: [Henning Wessels, Christian Weissenfels, Peter Wriggers]
year: 2020
venue: "CMAME (arXiv:2003.10208)"
gitrepo: "https://gitlab.com/henningwessels/npm"
---

## TL;DR
NPM is a Lagrangian-particle PINN for incompressible inviscid Euler with free surfaces. A network takes particle positions x_n and outputs s Implicit-Runge-Kutta velocity stages, pressure stages, and v_{n+1}. The incompressibility constraint is added as an extra squared-residual term — no LBB stabilization needed.

## Problem
Eulerian mesh PINNs for incompressible flow need pressure stabilization (PSPG, FIC) due to LBB/inf-sup condition and special techniques (VoF) for free surfaces. Eulerian convection terms also need stabilization. Need a meshfree, large-deformation friendly PINN that exactly satisfies div v = 0.

## Method
Updated Lagrangian: at each step the configuration at t_n is the reference; spatial derivatives are computed at x_n and pushed forward through the incremental deformation gradient Delta F_{n+1} = dx_{n+1}/dx_n.

For s-stage IRK, the network u_theta(x_n) outputs {v_i, p_i}_{i=1..s} and v_{n+1}. Position stages and divergence:
$$
x_i = x_n + \Delta t\sum_j a_{ij} v_j,\quad \Delta\dot{F}_i = \partial v_i/\partial x_n,\quad
\mathrm{div}\,v_i = \mathrm{tr}(\Delta\dot{F}_i \Delta F_i^{-1})
$$

Loss = momentum residual + incompressibility + boundary:
$$
SSE = \sum_j \|v_n^{(j)} - v_n\|^2 + \sum_i (\mathrm{div}\,v_i)^2 + |\mathrm{div}\,v_{n+1}|^2 + SSE_{BC}
$$
where v_n^{(j)} = v_j - Delta_t sum_i a_{ji} a(v_i, x_i) and a(v,x) = -(1/rho) grad p + b.

Exact BCs via Berg-Nystrom projection: u(x) = G(x) + D(x) u_hat(x) with D=0 on Gamma_D, G the smooth boundary extension. Pressure at t_{n+1} via Butcher weights: p_{n+1} = sum_j b_j p_j.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class NPMNet(nn.Module):
    d: int = 2
    s: int = 4
    width: int = 20
    depth: int = 2
    @nn.compact
    def __call__(self, x):
        h = x
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width)(h))
        out = nn.Dense(self.s*self.d + self.d + self.s)(h)
        vs       = out[..., :self.s*self.d].reshape(*out.shape[:-1], self.s, self.d)
        v_next   = out[..., self.s*self.d:self.s*self.d+self.d]
        p_stages = out[..., -self.s:]
        return vs, v_next, p_stages

def npm_loss(params, apply_fn, x_n, v_n, dt, a_tab, b_tab, rho, g, s, d):
    # Per-particle Jacobians of velocity and pressure stages w.r.t. x_n.
    def vs_fn(x):  return apply_fn(params, x)[0]   # (s,d)
    def ps_fn(x):  return apply_fn(params, x)[2]   # (s,)
    def per_point(xn):
        vs = vs_fn(xn)                              # (s,d)
        ps = ps_fn(xn)                              # (s,)
        dvs = jax.jacrev(vs_fn)(xn)                 # (s,d,d)
        dps = jax.jacrev(ps_fn)(xn)                 # (s,d)
        I   = jnp.eye(d)
        # incremental F_i = I + dt * sum_j a_ij * dvj/dxn
        F_i = I[None] + dt * jnp.einsum('ij,jab->iab', a_tab, dvs)   # (s,d,d)
        Finv = jnp.linalg.inv(F_i)                                   # (s,d,d)
        div_vi = jnp.einsum('iab,iba->i', dvs, Finv)                 # tr(dv * F^-1)
        grad_pi_curr = jnp.einsum('ia,iab->ib', dps, Finv)           # (s,d)
        a_i = -grad_pi_curr / rho + g                                # (s,d)
        v_n_pred = vs - dt * jnp.einsum('ij,jd->id', a_tab, a_i)     # (s,d)
        return ((v_n_pred - vn_target(xn))**2).sum() + (div_vi**2).sum()
    return jnp.mean(jax.vmap(per_point)(x_n))

net = NPMNet()
params = net.init(jax.random.PRNGKey(0), jnp.zeros(2))
opt = optax.adam(1e-3); state = opt.init(params)

@jax.jit
def step(params, state, x_n, v_n, dt, a_tab, b_tab, rho, g):
    grads = jax.grad(npm_loss)(params, net.apply, x_n, v_n,
                               dt, a_tab, b_tab, rho, g, 4, 2)
    upd, state = opt.update(grads, state, params)
    return optax.apply_updates(params, upd), state
```

Recommended: 2 hidden layers of 20 neurons, tanh, s=4 to 8 IRK stages, Adam then L-BFGS-B. Use Berg-Nystrom hard-BC (D analytic for rectangles, learn D with low-capacity ANN for complex shapes).

## Results
Sloshing in a container and 2-D dam break: NPM stays stable on highly irregular particle distributions, perfectly preserves the linear static pressure field over 50 time steps where a naive soft-BC PINN leaks at walls, and reproduces dam-break free surface evolution without any LBB stabilization.
