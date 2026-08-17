---
slot: 028
title: "A High-Efficient Hybrid Physics-Informed Neural Networks Based on Convolutional Neural Network"
authors: [Zhiwei Fang]
year: 2021
venue: "IEEE Transactions on Neural Networks and Learning Systems"
gitrepo: ""
doi: "10.1109/TNNLS.2021.3070878"
---

## TL;DR
Replace autodiff in PINN by a mesh-free finite-volume-style local-fitting stencil: for every point P_0 build a sparse linear combination w_i such that sum_i w_i u(P_i) approximates the differential operator exactly on a basis of polynomials. The resulting "hybrid PINN" applies the stencil like a CNN convolution and inherits a proven convergence rate.

## Problem
PINN's pointwise loss has too many degrees of freedom: even with MSE=0 in the limit, a network can satisfy the residual at sample points while taking arbitrary values in between (constant in a neighborhood, kink, etc.). VPINN/hp-VPINN suffer "variational crime" (trial=DNN does not match test space, LBB violated). No PINN variant had a numerical-analysis-style convergence rate.

## Method
For each point P_0 and its m local neighbors {P_i}, find weights w_i such that
$$
\mathcal{L}[u](P_0) \approx \sum_{i=0}^m w_i\,u(P_i)
$$
is exact for a chosen polynomial basis {p_1, ..., p_K} (e.g. for d=2, k=2 use {1, x, y, x^2, xy, y^2}, K=6). Solve the linear system
$$
\sum_i w_i p_j(P_i) = \mathcal{L}[p_j](P_0),\quad j=1,...,K
$$
via least squares (with hard constraints on the constant term to enforce consistency). The stencil convergence rate is O(h^{k - ord(L)}) where h is local point spacing.

Plug the stencil into the loss: replace AD-based residual N[u_theta](x) by
$$
\tilde N[u_\theta](x_i) = \sum_j w_{ij}\,u_\theta(x_{i,j})
$$
acting on the network's predictions at the stencil neighbors. Loss = MSE of residual + boundary MSE. Stencil is precomputed once.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from sklearn.neighbors import NearestNeighbors

def build_stencil(points, m=8, basis_deg=2, d=2):
    """Returns (N, m+1) int neighbour indices and (N, m+1) float weights."""
    nbrs = NearestNeighbors(n_neighbors=m+1).fit(points)
    _, idx = nbrs.kneighbors(points)
    polys = monomials_up_to(d, basis_deg)         # list of (p, L[p])
    weights = np.zeros((len(points), m+1))
    for i, ids in enumerate(idx):
        P = points[ids]                            # (m+1, d)
        A = np.stack([p(P) for p,_ in polys], axis=0)       # (K, m+1)
        b = np.array([Lp(points[i]) for _,Lp in polys])     # (K,)
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
        weights[i] = w
    return jnp.asarray(idx), jnp.asarray(weights)            # JAX arrays

class MLP(nn.Module):
    width: int = 50; depth: int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.width)(x))
        return nn.Dense(1)(x)

def hybrid_residual(params, apply_fn, x, idx, w, f):
    # u_all : (N, 1).  idx : (N, m+1).  w : (N, m+1).
    u_all = jax.vmap(lambda z: apply_fn(params, z)[0])(x)             # (N,)
    u_nbr = u_all[idx]                                                # (N, m+1)
    return jnp.sum(w * u_nbr, axis=-1) - jax.vmap(f)(x)               # (N,)

def loss_bc(params, apply_fn, x_bnd, g):
    return jnp.mean((jax.vmap(lambda z: apply_fn(params, z)[0])(x_bnd) - jax.vmap(g)(x_bnd))**2)

net = MLP()
params = net.init(jax.random.PRNGKey(0), jnp.zeros(2))
opt = optax.adam(1e-3); state = opt.init(params)
idx, w = build_stencil(np.asarray(x_interior), m=8, basis_deg=2)

@jax.jit
def step(params, state, x_interior, idx, w, x_bnd):
    def total(p):
        r  = hybrid_residual(p, net.apply, x_interior, idx, w, f_source)
        return jnp.mean(r**2) + loss_bc(p, net.apply, x_bnd, g_bc)
    grads = jax.grad(total)(params)
    upd, state = opt.update(grads, state, params)
    return optax.apply_updates(params, upd), state
```

Recommended: m=8 nearest neighbors in 2D (5 for axis-aligned), polynomial degree k = ord(L) + 1 or +2 for better order. Stencil weights are reusable — compute once.

## Results
On Poisson and advection-diffusion in 2-D, hybrid PINN reaches L2 errors comparable to AD-PINN but in ~5x less training time, and demonstrates the predicted O(h^p) convergence as points densify — first PINN variant with a guaranteed convergence rate. Extension to surface PDEs shown without proof.
