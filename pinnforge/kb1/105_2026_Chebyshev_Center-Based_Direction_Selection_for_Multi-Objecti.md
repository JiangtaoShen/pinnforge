---
slot: 105
title: "Chebyshev Center-Based Direction Selection for Multi-Objective Optimization and Training PINNs"
authors: [Hoyeol Yoon, Seoungbin Bae, Nam Ho-Nguyen, Dabeen Lee]
year: 2026
venue: arXiv:2605.09975
gitrepo: ""
---

## TL;DR
PINN training has m loss gradients `{g_i}` that often conflict. This paper picks the parameter-update direction as the Chebyshev center of the dual cone `K* = {v : g_i^T v ≥ 0 ∀i}`: the normalised direction that maximises the minimum distance to every cone facet. One geometric rule recovers scale robustness, balanced descent and simultaneous descent simultaneously, and is solvable through an m-dim simplex dual.

## Problem
PINN loss `L = L_r + L_b + L_i + ...` produces conflicting per-objective gradients. Existing direction-selection methods (DCGD, ConFIG, HARMONIC, IMTL-G, MGDA, CAGrad, PCGrad) each *impose* a desirable property (equalised progress, simultaneous descent, scale robustness). It is unclear which property is essential or how the methods are geometrically related.

## Method
Let `ĝ_i = g_i/‖g_i‖_p`. At each step solve the primal Chebyshev-center problem in parameter space `R^n`:
$$
\max_{v,\,r}\; r\quad \text{s.t.}\quad \hat g_i^\top v \ge r\ \forall i,\ \|v\|_q \le 1,\qquad q=p/(p-1).
$$
By LP duality this is equivalent to the convex simplex problem in `R^m`:
$$
\min_{\alpha\in\Delta_m}\;\Big\|\sum_{i=1}^m \alpha_i\, \hat g_i\Big\|_p.
$$
For `1<p<∞`, the primal direction is recovered from `w^\star=\sum_i \alpha_i^\star \hat g_i` as
$$
v^\star = \operatorname{sgn}(w^\star)\odot |w^\star|^{p-1}\big/\|w^\star\|_p^{p-1}.
$$
For `p=q=2`, `v^\star = w^\star/\|w^\star\|`. The final update direction is `d_t = (\sum_i g_i^\top v^\star)\,v^\star` (the inner-product scalar inherits magnitude). Pareto-stationarity holds when the optimal radius `r^\star=0`, giving a stopping criterion. With `L = ∑ L_i` being β-smooth in ‖·‖_q, the step size η=1/β yields O(1/T) convergence to ε-Pareto-stationary points.

```python
import jax, jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

def flat_grad(loss_fn, params, *args):
    g = jax.grad(loss_fn)(params, *args)
    return jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(g)])

def chebyshev_dir(loss_fns, params, args_list, p=2):
    G = jnp.stack([flat_grad(L, params, *a) for L, a in zip(loss_fns, args_list)])
    Gn = G / (jnp.linalg.norm(G, ord=p, axis=1, keepdims=True) + 1e-12)
    m = G.shape[0]
    Gn_np = np.asarray(Gn)
    def obj(a): return float(np.sum((a[:, None] * Gn_np).sum(0) ** p) ** (1.0/p))
    a0 = np.full((m,), 1.0/m)
    cons = ({'type': 'eq', 'fun': lambda a: a.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * m
    res = minimize(obj, a0, bounds=bnds, constraints=cons, method='SLSQP')
    alpha = jnp.asarray(res.x, dtype=G.dtype)
    w = (alpha[:, None] * Gn).sum(0)
    if p == 2:
        v = w / (jnp.linalg.norm(w) + 1e-12)
    else:
        v = jnp.sign(w) * jnp.abs(w)**(p-1) / (jnp.linalg.norm(w, ord=p)**(p-1) + 1e-12)
    d = (G @ v).sum() * v
    return d

# training step: apply -lr*d to flat params, then unravel back
from jax.flatten_util import ravel_pytree
flat_params, unravel = ravel_pytree(params)
d = chebyshev_dir([loss_r, loss_b, loss_i], params,
                  [(x_r,), (x_b, u_b), (x_i, u_i)])
flat_new = flat_params - 1e-3 * d
params = unravel(flat_new)
```

Defaults: p=2 (Euclidean), Frank-Wolfe or SLSQP for the m-dim simplex, ε≈1e-6 stopping tol, η~1e-3 with Adam-style schedule. The dual is m-dim (3-5 losses for typical PINNs) so per-step overhead is negligible.

## Results
Across PINN benchmarks (Burgers, Helmholtz, KdV, Navier-Stokes), the method matches or beats DCGD, HARMONIC, ConFIG, IMTL-G, MGDA, GAPO in relative L² error and converges more reliably than scale-sensitive baselines. The single geometric principle subsumes the union of properties prior cone-based methods enforce one by one.
