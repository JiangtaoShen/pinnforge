---
slot: 035
title: "Efficient training of physics-informed neural networks via importance sampling"
authors: [Mohammad Amin Nabian, Rini Jasmine Gladstone, Hadi Meidani]
year: 2021
venue: "Computer-Aided Civil and Infrastructure Engineering (arXiv:2104.12325)"
gitrepo: ""
---

## TL;DR
Sample collocation points each iteration proportionally to the per-point PINN loss (importance sampling), with the Horvitz-Thompson 1/q correction to keep the gradient unbiased. Add a "piecewise-constant" approximation: evaluate loss only on a coarse seed set, propagate to neighbors via Voronoi/nearest-neighbor — no new hyperparameters and a few-line code change.

## Problem
Uniform collocation wastes gradient steps on regions where the residual is already small. Sampling proportional to per-sample gradient norm is theoretically optimal but requires an extra backward pass per point. Need a cheap surrogate that still accelerates convergence.

## Method
For loss J(theta) = E_f [J(theta; x)] over x ~ f, draw x ~ q instead and correct:
$$
\theta^{i+1} = \theta^i - \frac{\eta}{m N}\sum_{j\in\mathcal{M}^i}\frac{1}{q_j^i}\nabla_\theta J(\theta^i; x_j)
$$
Optimal q* prop to ||grad_theta J||, but theorem from Katharopoulos & Fleuret: q_j prop to J(theta; x_j) is an upper bound and preserves the ranking. So use
$$
q_j^i = \frac{J(\theta^i; x_j)}{\sum_{k=1}^N J(\theta^i; x_k)}
$$

PWC speedup: choose S << N seed points {x_s}; build Voronoi/k-NN partition. Set q_j = J(theta; x_{rho(j)}) where rho(j) is the seed nearest to x_j. Only S residual evaluations per refresh instead of N.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from sklearn.neighbors import NearestNeighbors

def importance_sampling_pinn(net, params, pde_loss_per_point,
                             x_all, seeds_idx, key,
                             m=256, n_iter=100000, lr=1e-3):
    """
    x_all      : (N, d) pre-sampled candidate collocation points.
    seeds_idx  : (S,) indices into x_all that act as PWC seeds.
    """
    N = x_all.shape[0]
    nbrs = NearestNeighbors(n_neighbors=1).fit(jax.device_get(x_all[seeds_idx]))
    _, rho_np = nbrs.kneighbors(jax.device_get(x_all))
    rho = jnp.asarray(rho_np.flatten())                   # rho[j] -> seed index

    opt = optax.adam(lr); state = opt.init(params)
    apply_fn = net.apply

    @jax.jit
    def refresh_q(params):
        J_seeds = jax.vmap(lambda x: pde_loss_per_point(params, apply_fn, x))(
            x_all[seeds_idx])                             # (S,)
        J_seeds = jax.lax.stop_gradient(J_seeds)
        J_per_pt = J_seeds[rho]                           # (N,)
        return J_per_pt / (jnp.sum(J_per_pt) + 1e-12)

    @jax.jit
    def step(params, state, idx, q):
        x_batch = x_all[idx]
        def loss(p):
            J = jax.vmap(lambda x: pde_loss_per_point(p, apply_fn, x))(x_batch)  # (m,)
            w = 1.0 / (N * q[idx])
            return jnp.mean(w * J)
        g = jax.grad(loss)(params)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state

    for it in range(n_iter):
        q = refresh_q(params)
        key, sub = jax.random.split(key)
        idx = jax.random.choice(sub, N, shape=(m,), replace=True, p=q)
        params, state = step(params, state, idx, q)
    return params
```

Recommended: N = 10^4-10^5 candidate points (uniform or Halton/Sobol low-discrepancy), S ~ 200-500 seeds (refresh seed losses every step or every K iters), m = 128-256, Sine activation, 4x32 net for elasticity. Use lambda~1-100 for BC terms.

## Results
On 2-D elasticity over an irregular plate, plane-stress with 3 holes, and steady diffusion: 1.5-3x faster convergence vs uniform sampling at equal compute, lower variance across seeds. PWC approximation matches per-point importance sampling at a fraction of the cost.
