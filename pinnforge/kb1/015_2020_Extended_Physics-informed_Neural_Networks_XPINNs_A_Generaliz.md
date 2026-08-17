---
slot: 15
title: "Extended Physics-informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition based Deep Learning Framework for Nonlinear Partial Differential Equations"
authors: [Ameya D. Jagtap, George Em Karniadakis]
year: 2020
venue: "AAAI Spring Symposium: MLPS"
gitrepo: "https://github.com/AmeyaJagtap/XPINNs"
---

## TL;DR
Generalisation of cPINN to *any* PDE. Decompose `Omega` (and time) into arbitrarily-shaped sub-domains (convex or non-convex); one sub-PINN per subdomain. At interfaces enforce two simple terms: (i) *residual continuity* (the PDE residual evaluated by both adjacent sub-nets must match) and (ii) *average-solution* matching. No normal-direction / flux computation, so works for non-conservation-law PDEs (Poisson, Helmholtz, biharmonic, etc.) and irregular interfaces (cracks, moving fronts).

## Problem
cPINN couples sub-domains by *flux continuity*, requiring a conservation form and explicit normal direction at interfaces - impractical for non-conservation PDEs and irregular/moving interfaces. We need a generic, interface-shape-agnostic stitching mechanism.

## Method
Subdomains `Omega_q`, `q = 1..N_sd`, non-overlapping, intersecting only on `dOmega_{ij}`. Sub-Net `u_{theta_q}` with locally-adaptive activations (`sigma(n a_k z)`, `n=5`, `a_k` trainable per layer). On the common interface a point can belong to `S` subdomains; the network output there is averaged: `u(z) = (1/S) sum_q u_{theta_q}(z)`.

Per-subdomain loss:
$$
J(\tilde\Theta_q) = W_u^q\,\mathrm{MSE}_u^q + W_F^q\,\mathrm{MSE}_F^q + W_I^q\,\mathrm{MSE}_{u_{avg}}^q + W_{IF}^q\,\mathrm{MSE}_R^q
$$
where:
- `MSE_F^q = (1/N_F) sum |F(u_{theta_q}; lambda)|^2` -- PDE residual interior to `Omega_q`.
- `MSE_u^q = (1/N_u) sum |u_{theta_q}(x_u) - u^*|^2` -- data / IC / BC.
- `MSE_{u_avg}^q = (1/N_I) sum |u_{theta_q}(x_I) - (1/S) sum_p u_{theta_p}(x_I)|^2`. (sub-net agrees with the average)
- `MSE_R^q = (1/N_I) sum |F_q(u_{theta_q}(x_I)) - F_p(u_{theta_p}(x_I))|^2` -- *residual continuity* across the interface. This is the key XPINN ingredient: at every interface point, the *PDE residual* evaluated by both sides must agree. No flux, no normals.

Optionally also enforce `u_{theta_q}(x_I) = u_{theta_p}(x_I)` directly (`C^0` matching).

Slope-recovery regulariser `S(a)` same as cPINN: `1 / mean(exp(a_k))`.

JAX (key interface routine, flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class AdaptiveMLP(nn.Module):
    h: int = 20
    depth: int = 4
    out_dim: int = 1
    n: float = 5.0
    @nn.compact
    def __call__(self, x):
        a = self.param("a", lambda key: jnp.full((self.depth,), 1.0 / self.n))
        for k in range(self.depth):
            x = jnp.tanh(self.n * a[k] * nn.Dense(self.h)(x))
        return nn.Dense(self.out_dim)(x)

def pde_residual(net, p, xt, lam):              # viscous Burgers
    def u_single(pp, t, x): return net.apply(pp, jnp.array([[t, x]]))[0, 0]
    u    = jax.vmap(lambda t, x: u_single(p, t, x))(xt[:, 0], xt[:, 1])
    u_t  = jax.vmap(lambda t, x: jax.grad(u_single, 1)(p, t, x))(xt[:, 0], xt[:, 1])
    u_x  = jax.vmap(lambda t, x: jax.grad(u_single, 2)(p, t, x))(xt[:, 0], xt[:, 1])
    u_xx = jax.vmap(lambda t, x: jax.grad(jax.grad(u_single, 2), 2)(p, t, x))(xt[:, 0], xt[:, 1])
    return u_t + u * u_x - lam * u_xx, u

nets   = [AdaptiveMLP() for _ in range(N_sd)]
keys   = jax.random.split(jax.random.PRNGKey(0), N_sd)
params = [n.init(k, jnp.zeros((1, 2))) for n, k in zip(nets, keys)]
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

def xpinn_loss(params, batches, lam):
    L_tot = 0.0
    for q in range(N_sd):
        R_q, u_q = pde_residual(nets[q], params[q], batches["int"][q], lam)
        L_F = jnp.mean(R_q ** 2)
        L_u = jnp.mean((nets[q].apply(params[q], batches["data"][q]) - batches["ud"][q]) ** 2)
        L_tot = L_tot + W_F * L_F + W_u * L_u
    for (p, q), x_I in batches["interface"].items():
        R_p, u_p = pde_residual(nets[p], params[p], x_I, lam)
        R_q, u_q = pde_residual(nets[q], params[q], x_I, lam)
        u_avg = 0.5 * (u_p + u_q)
        L_avg = jnp.mean((u_p - u_avg) ** 2) + jnp.mean((u_q - u_avg) ** 2)
        L_R   = jnp.mean((R_p - R_q) ** 2)
        L_tot = L_tot + W_I * L_avg + W_IF * L_R
    return L_tot

@jax.jit
def train_step(params, opt_state, batches, lam):
    grads = jax.grad(xpinn_loss)(params, batches, lam)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

Recommended: `W_F : W_u : W_I : W_IF = 1 : 1 : 20 : 1`; 4-64 subdomains depending on geometry; depth/width per subdomain chosen by solution regularity (shallow `[2,20,20,1]` in smooth zones, deeper in shocks); `n=5`; Adam then L-BFGS.

## Results
Demonstrated on 1-D/2-D Burgers, 2-D Helmholtz, 2-D Poisson on `X`-shaped and L-shaped domains, 3-D Poisson, and inverse problems with piecewise parameters. Matches/exceeds cPINN accuracy on conservation laws (relative L2 ~ 1e-3 to 1e-4) and works on non-conservation PDEs where cPINN is inapplicable; parallelises naturally across subdomains.
