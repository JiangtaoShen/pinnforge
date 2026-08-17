---
slot: 032
title: "CAN-PINN: A Fast Physics-Informed Neural Network Based on Coupled-Automatic-Numerical Differentiation Method"
authors: [Pao-Hsiung Chiu, Jian Cheng Wong, Chinchun Ooi, My Ha Dao, Yew-Soon Ong]
year: 2021
venue: "Computer Methods in Applied Mechanics and Engineering"
gitrepo: ""
---

## TL;DR
PINN's pure-AD residual can be fit pointwise without actually solving the PDE (under-constrained in sparse-collocation regime). CAN-PINN couples AD with a numerical-difference stencil on neighbor support points: residual is a Taylor-consistent finite-difference formula whose endpoint slopes come from AD. Adds inter-point coupling, drops the collocation budget by orders of magnitude.

## Problem
With insufficient collocation points, a-PINN (auto-diff PINN) trains its residual to ~1e-7 yet returns a solution unrelated to the truth — degrees of freedom in the DNN > constraints at isolated points. n-PINN (finite-difference) couples neighbors but inherits the stencil's truncation error.

## Method
For each collocation x_i and chosen spacing Delta x, evaluate the network at x_i, x_i +/- Delta x as "support points". AD computes u_x at those points; combine them via Taylor expansion.

A. **can(uw2)** upwind first-derivative for convection ut + a ux = 0 (a>0):
$$
u_x|_{can,uw2} = \tfrac{1}{\Delta x}\big[u_\theta(x_i) - u_\theta(x_i-\Delta x)\big] + \tfrac{\Delta x}{2}\big[u_{\theta,x}(x_i) - u_{\theta,x}(x_i-\Delta x)\big]
$$
2nd-order accurate, adds dispersion stabilization.

B. **can(cd)** central scheme for pressure gradient in incompressible NS:
$$
p_x|_{can,cd} = \tfrac{1}{2\Delta x}[p_\theta(x_i+\Delta x) - p_\theta(x_i-\Delta x)] + \tfrac{\Delta x}{6}[p_\theta(x_i+\Delta x) - 2p_\theta(x_i) + p_\theta(x_i-\Delta x)]
$$
Avoids velocity-pressure decoupling on collocated grids; non-dissipative.

As Delta x -> 0 these recover AD exactly. Loss uses the can-residual instead of the AD residual; everything else (BC/IC MSE, Adam) unchanged.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax, math

class SineMLP(nn.Module):
    """Sinusoidal first-layer features (sigma=1), sine hidden activations."""
    width: int = 50; depth: int = 4; sigma: float = 1.0
    @nn.compact
    def __call__(self, x):
        W = self.param("W0", nn.initializers.normal(self.sigma), (x.shape[-1], self.width))
        h = jnp.sin(2.0 * math.pi * (x @ W))
        for _ in range(self.depth - 1):
            h = jnp.sin(nn.Dense(self.width)(h))
        return nn.Dense(1)(h)

def can_uw2_residual_burgers(params, apply_fn, x, nu, dx, a=1.0):
    """ut + a*ux - nu*uxx = 0 via can(uw2) on ux, central FD on uxx."""
    def u_fn(z): return apply_fn(params, z)[0]
    e = jnp.zeros_like(x); e = e.at[0].set(dx)
    u  = u_fn(x);  u_m = u_fn(x - e);  u_p = u_fn(x + e)
    ut    = jax.grad(u_fn)(x)[1]
    ux_AD = jax.grad(u_fn)(x)[0]
    ux_m  = jax.grad(u_fn)(x - e)[0]
    ux_can  = (u - u_m)/dx + 0.5 * (ux_AD - ux_m)
    uxx_can = (u_p - 2*u + u_m) / dx**2
    return ut + a*ux_can - nu*uxx_can

def loss_can(params, apply_fn, x_r, x_b, x_ic, u_ic, nu, dx):
    r   = jax.vmap(lambda z: can_uw2_residual_burgers(params, apply_fn, z, nu, dx))(x_r)
    L_R = jnp.mean(r**2)
    L_B = jnp.mean(jax.vmap(lambda z: apply_fn(params, z)[0])(x_b)**2)
    L_I = jnp.mean((jax.vmap(lambda z: apply_fn(params, z)[0])(x_ic) - u_ic)**2)
    return L_R + L_B + L_I

net = SineMLP()
params = net.init(jax.random.PRNGKey(0), jnp.zeros(2))
opt = optax.adam(1e-3); state = opt.init(params)

@jax.jit
def step(params, state, x_r, x_b, x_ic, u_ic, nu, dx):
    g = jax.grad(loss_can)(params, net.apply, x_r, x_b, x_ic, u_ic, nu, dx)
    upd, state = opt.update(g, state, params)
    return optax.apply_updates(params, upd), state
```

Recommended: sinusoidal first-layer features (sigma=1), sine activations in hidden layers, separate output sub-nets for (u,v,p) in NS, Delta x ~ 0.02 in physical units (problem-dependent), Adam with ReduceLROnPlateau.

## Results
On ODE u'' = f with only 41 points, a-PINN MSE ~1e-1 (train loss ~1e-7 but wrong solution); can-PINN MSE ~1e-5. On flow-mixing, lid-driven cavity (Re=100), backward-facing step, can(uw2)+can(cd) consistently 1-2 orders of magnitude better than n-PINN and recovers solutions where a-PINN fails. Successfully infers Reynolds number from sparse observations.
