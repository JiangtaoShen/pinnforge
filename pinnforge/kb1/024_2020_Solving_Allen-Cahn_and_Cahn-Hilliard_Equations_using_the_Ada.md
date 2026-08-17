---
slot: 024
title: "Solving Allen-Cahn and Cahn-Hilliard Equations using the Adaptive Physics Informed Neural Networks"
authors: [Colby L. Wight, Jia Zhao]
year: 2020
venue: "Communications in Computational Physics (arXiv:2007.04542)"
gitrepo: ""
---

## TL;DR
For phase-field PDEs with sharp moving interfaces (Allen-Cahn, Cahn-Hilliard), vanilla PINN fails. The paper combines IC weight C0~100, mini-batched Adam, residual-driven adaptive collocation resampling, and two time-adaptive strategies (gradually extending the time window vs. sequential time-marching with separate networks).

## Problem
Allen-Cahn ut = gamma1 Delta u + gamma2(u - u^3) and Cahn-Hilliard ut = Delta(-gamma1 Delta u + gamma2(u^3-u)) have steep interfaces that move in time. Fixed collocation points (LHS at init) leave too few points near the interface; baseline PINN gets stuck at L2 ~ O(1) on AC and diverges on CH due to the 4th-order Laplacian-squared.

## Method

A. Loss with heavy IC weight (time-irreversibility):
$$
\mathrm{MSE} = C_0\,\mathrm{MSE}_u + \mathrm{MSE}_b + \mathrm{MSE}_f,\quad C_0 \approx 100
$$
MSE_u: |U(0,x_i) - u_i|^2; MSE_b: BC residual; MSE_f: PDE residual via autograd. Mini-batches over collocation points (NOT full batch) escape bad minima.

B. Adaptive spatial resampling on residual. Every K epochs, sample a large LHS pool, evaluate |f| on it, add the top-M points (highest |f|) to the collocation set; keep old points (cumulative). The f-network error is the indicator.

C. Time-adaptive I (single network, growing window). Train only on collocation points with t in [0, t_i] for ever-growing t_i = 0.1, 0.2, ..., 1.0. Move to next window when MSE_f < tol or max-iters hit. Older points stay in the pool.

D. Time-adaptive II (time-marching, multiple networks). Split [0,T] into chunks of length dt~0.1. Train net_k on [t_{k-1}, t_k] using net_{k-1}'s prediction at t = t_{k-1} as IC. Networks combine into a piecewise solution.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from scipy.stats.qmc import LatinHypercube

class PINN(nn.Module):
    width: int = 128
    depth: int = 8
    @nn.compact
    def __call__(self, xt):
        h = xt
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width)(h))
        return nn.Dense(1)(h)

def ac_residual_point(params, apply_fn, xt, gamma1, gamma2):
    def u_fn(z): return apply_fn(params, z)[0]
    g    = jax.grad(u_fn)(xt)             # (2,) -> [dx, dt]
    H    = jax.hessian(u_fn)(xt)          # (2,2)
    u    = u_fn(xt); ut = g[1]; uxx = H[0,0]
    return ut - gamma1*uxx - gamma2*(u - u**3)

def losses(params, apply_fn, batch_coll, x0, u0, xb, g1, g2, C0):
    f = jax.vmap(lambda z: ac_residual_point(params, apply_fn, z, g1, g2))(batch_coll)
    Lf = jnp.mean(f**2)
    Lu = jnp.mean((jax.vmap(lambda z: apply_fn(params, z)[0])(x0) - u0)**2)
    Lb = jnp.mean(jax.vmap(lambda z: apply_fn(params, z)[0])(xb)**2)
    return C0*Lu + Lb + Lf

def train_with_adaptive(params, apply_fn, x0, u0, xb, x_coll,
                        K=2000, M=500, C0=100.0,
                        num_resamples=10, g1=1e-4, g2=5.0):
    opt = optax.adam(1e-3); state = opt.init(params)

    @jax.jit
    def step(params, state, batch):
        grads = jax.grad(losses)(params, apply_fn, batch, x0, u0, xb, g1, g2, C0)
        upd, state = opt.update(grads, state, params)
        return optax.apply_updates(params, upd), state

    for outer in range(num_resamples):
        for k in range(K):
            for batch in minibatches(x_coll, bs=256):
                params, state = step(params, state, batch)
        # adaptive resampling: rank LHS pool by |f|
        pool = lhs_sample(N=20000)
        f_pool = jnp.abs(jax.vmap(
            lambda z: ac_residual_point(params, apply_fn, z, g1, g2))(pool))
        idx = jnp.argsort(-f_pool)[:M]
        x_coll = jnp.concatenate([x_coll, pool[idx]], axis=0)
    return params

# Time-marching variant
for k in range(K_chunks):
    net_k = PINN()
    params_k = net_k.init(jax.random.PRNGKey(k), jnp.zeros(2))
    x_coll_k = lhs_sample_in_window(t_lo=k*dt, t_hi=(k+1)*dt)
    # warm IC: evaluate previous network on the chunk-start grid (no-grad)
    u_ic = jax.lax.stop_gradient(
        jax.vmap(lambda x: prev_apply(prev_params, x))(grid_at_t_kdt))
    params_k = train_with_adaptive(params_k, net_k.apply, grid, u_ic, xb, x_coll_k)
```

Recommended: 8 layers x 128 units, tanh, Adam 1e-3, then L-BFGS. C0=100, resample every ~2000 epochs, M=500 added points per round, dt=0.1 for time-marching.

## Results
On AC (gamma2=5) baseline PINN error ~0.5; adaptive sampling drops it to ~5e-3, and time-marching to ~2e-3. CH 4th-order PDE solvable only with time-marching, achieving relative L2 ~1e-2.
