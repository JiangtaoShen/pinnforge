---
slot: 029
title: "A Physics Informed Neural Network for Time-Dependent Nonlinear and Higher Order Partial Differential Equations"
authors: [Revanth Mattey, Susanta Ghosh]
year: 2021
venue: "CMAME (arXiv:2106.07606)"
gitrepo: ""
---

## TL;DR
bc-PINN: train a SINGLE network sequentially over short time segments. Each new segment uses standard PINN loss PLUS a "backward compatibility" MSE that penalizes departure from the already-learned solution on all prior segments. Yields the accuracy of time-marching with the smoothness of a single global net — solves Allen-Cahn (strong nonlinearity) and Cahn-Hilliard (4th-order) where vanilla PINN fails.

## Problem
Vanilla PINN can't fit strongly nonlinear PDEs like AC (f(h)=5(h^3-h)) or 4th-order CH — the nonlinear term residual is mis-learned and never recovered. Naive time-marching with separate nets discards information and gives discontinuities at chunk boundaries.

## Method
Discretize time [0,T] into n_max segments [T_{k-1}, T_k]. Train ONE network h_theta(x,t) on segment k by minimizing
$$
\mathrm{MSE}_{\Delta T_k} = \mathrm{MSE}_I + \mathrm{MSE}_B + \mathrm{MSE}_R + \mathrm{MSE}_S\quad (k\ge 2)
$$
The new term MSE_S enforces backward compatibility on a stored set of points (x_s, t_s) in Omega x [0, T_{k-1}] (the past):
$$
\mathrm{MSE}_S = \tfrac{1}{N_s}\sum_{j=1}^{N_s} \big(h_\theta(x_s^j, t_s^j) - \tilde h(x_s^j, t_s^j)\big)^2
$$
where tilde_h is the FROZEN copy of the network at the end of the previous segment, evaluated once per outer step (or pre-evaluated to numerical values stored on a grid).

For higher-order PDEs (e.g. Cahn-Hilliard with Delta^2 h), introduce a phase-space split: predict (h, w) with auxiliary network output w approximating Delta h, then enforce w - Delta h = 0 and ht - Delta(-c1^2 w + c2(h^3 - h)) = 0 — lowers max derivative order to 2 and stabilizes training.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class BCPINN(nn.Module):
    width: int = 200; depth: int = 4; d_out: int = 1
    @nn.compact
    def __call__(self, xt):
        for _ in range(self.depth):
            xt = nn.tanh(nn.Dense(self.width)(xt))
        return nn.Dense(self.d_out)(xt)

def ac_residual_point(params, apply_fn, xt, c1sq=1e-4, c2=5.0):
    def h_fn(z): return apply_fn(params, z)[0]
    g  = jax.grad(h_fn)(xt)
    H  = jax.hessian(h_fn)(xt)
    h  = h_fn(xt); ht = g[1]; hxx = H[0,0]
    return ht - c1sq*hxx + c2*(h**3 - h)

def total_loss(params, apply_fn, xt_r, xt_b, x_ic, h_ic,
               xt_past, h_past_frozen, include_ic):
    r = jax.vmap(lambda z: ac_residual_point(params, apply_fn, z))(xt_r)
    L_R = jnp.mean(r**2)
    L_B = bc_loss(params, apply_fn, xt_b)        # e.g. periodic mismatch
    L_I = jnp.where(include_ic,
                    jnp.mean((jax.vmap(lambda z: apply_fn(params, z)[0])(x_ic) - h_ic)**2),
                    0.0)
    L_S = jnp.where(xt_past.shape[0] > 0,
                    jnp.mean((jax.vmap(lambda z: apply_fn(params, z)[0])(xt_past)
                              - h_past_frozen)**2),
                    0.0)
    return L_R + L_B + L_I + L_S

def train_bc_pinn(T_segments, x_ic, h_ic):
    net = BCPINN()
    params = net.init(jax.random.PRNGKey(0), jnp.zeros(2))
    opt = optax.adam(1e-3); state = opt.init(params)
    xt_past = jnp.empty((0,2)); h_past = jnp.empty((0,))

    @jax.jit
    def step(params, state, xt_r, xt_b, x_ic, h_ic,
             xt_past, h_past, include_ic):
        grads = jax.grad(total_loss)(params, net.apply, xt_r, xt_b,
                                     x_ic, h_ic, xt_past, h_past, include_ic)
        upd, state = opt.update(grads, state, params)
        return optax.apply_updates(params, upd), state

    for k, (t_lo, t_hi) in enumerate(T_segments):
        xt_r = sample_coll(t_lo, t_hi); xt_b = sample_bc(t_lo, t_hi)
        for it in range(N_ADAM):
            params, state = step(params, state, xt_r, xt_b,
                                 x_ic, h_ic, xt_past, h_past, jnp.array(k == 0))
        # Snapshot at end of segment for future MSE_S.
        params_frozen = jax.tree_util.tree_map(lambda x: x, params)  # copy
        xt_grid  = sample_grid(0.0, t_hi)
        h_frozen = jax.lax.stop_gradient(
            jax.vmap(lambda z: net.apply(params_frozen, z)[0])(xt_grid))
        xt_past, h_past = xt_grid, h_frozen
    return params
```

Recommended: 4 hidden x 200 tanh, Xavier init, Adam(1e-3) -> L-BFGS, ~20k collocation/segment, dt ~ 0.05 to 0.1, N_s ~ 5000 stored points.

## Results
On AC (c1^2=1e-4, c2=5): vanilla PINN err ~ 0.7 (nonlinear term wrong); bc-PINN ~ 5e-3. On CH 4th-order with phase-space split: bc-PINN ~ 1e-2 where vanilla diverges. Uses fewer collocation points and Adam iterations than baseline PINN.
