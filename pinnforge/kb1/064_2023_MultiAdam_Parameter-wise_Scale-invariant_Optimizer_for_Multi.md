---
slot: 64
title: "MultiAdam: Parameter-wise Scale-invariant Optimizer for Multiscale Training of PINNs"
authors: [Jiachen Yao, Chang Su, Zhongkai Hao, Songming Liu, Hang Su, Jun Zhu]
year: 2023
venue: ICML 2023 (arXiv:2306.02816)
gitrepo: "https://github.com/i207M/MultiAdam"
---

## TL;DR
For a k-th-order PDE, shrinking the domain by `t` multiplies the PDE loss by `t^{2k}` while the boundary loss stays put — vanilla Adam silently lets the dominant term win. **MultiAdam** runs one Adam state per loss group (each PDE residual one group; all BC/IC losses one group), normalizes each group's gradient by its own second moment, then averages — making the update **scale-invariant** at the parameter level.

## Problem
On a 1x1 unit domain, the Poisson PDE loss can be 8^4 ≈ 4000x larger than its BC loss; Adam, LRA, NTK, GradNorm, PCGrad all degrade or fail when the domain scale changes. The total weighted loss is also a poor proxy for actual L2 error — minimizing it does not imply better predictions until a per-group scale-invariant bound is enforced.

## Method
Split the PINN total loss into `n` groups `{f_1, ..., f_n}`: one group per distinct PDE residual (each scales differently because of derivative orders), one group for all Dirichlet/Neumann BC + IC losses (these are pure L2 on samples, scale-invariant).

For each group `i` maintain independent Adam moments `m_i, v_i`. Each step:
$$
g_{t,i} = \nabla_\theta f_i(\theta_{t-1}),\quad
m_{t,i} = \beta_1 m_{t-1,i} + (1-\beta_1) g_{t,i},\quad
v_{t,i} = \beta_2 v_{t-1,i} + (1-\beta_2) g_{t,i}^2
$$
$$
\hat m_{t,i} = m_{t,i}/(1-\beta_1^t),\quad \hat v_{t,i} = v_{t,i}/(1-\beta_2^t)
$$
$$
\theta_t = \theta_{t-1} - \frac{\gamma}{n}\sum_{i=1}^{n} \frac{\hat m_{t,i}}{\sqrt{\hat v_{t,i}} + \varepsilon}
$$

`sqrt(v)` is the per-group, per-parameter gradient magnitude estimate; dividing by it normalizes each group's contribution to the same scale before averaging. The average (not sum) keeps the overall step size like a single Adam.

Recommended hyper-params (from sensitivity study Appendix D):
- `gamma = 1e-3`, `beta_1 = 0.99`, `beta_2 = 0.99` (high momentum stabilizes the per-group `v` estimates).
- `eps = 1e-8`.
- For coupled systems, give each conservation law its own group.

```python
import jax, jax.numpy as jnp

def init_multiadam(params, n_groups):
    zeros = lambda: jax.tree_util.tree_map(jnp.zeros_like, params)
    return {
        "m":    [zeros() for _ in range(n_groups)],
        "v":    [zeros() for _ in range(n_groups)],
        "step": 0,
    }

def multiadam_update(params, state, loss_fns, args,
                     lr=1e-3, b1=0.99, b2=0.99, eps=1e-8):
    n = len(loss_fns)
    grads_per_group = [jax.grad(lf)(params, *args) for lf in loss_fns]
    new_m, new_v = [], []
    update = jax.tree_util.tree_map(jnp.zeros_like, params)
    t = state["step"] + 1
    for i in range(n):
        g  = grads_per_group[i]
        mi = jax.tree_util.tree_map(lambda mm, gg: b1 * mm + (1 - b1) * gg,
                                    state["m"][i], g)
        vi = jax.tree_util.tree_map(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg,
                                    state["v"][i], g)
        new_m.append(mi); new_v.append(vi)
        m_hat = jax.tree_util.tree_map(lambda x: x / (1 - b1**t), mi)
        v_hat = jax.tree_util.tree_map(lambda x: x / (1 - b2**t), vi)
        delta = jax.tree_util.tree_map(lambda m, v: m / (jnp.sqrt(v) + eps),
                                       m_hat, v_hat)
        update = jax.tree_util.tree_map(jnp.add, update, delta)
    new_params = jax.tree_util.tree_map(lambda p, u: p - (lr / n) * u,
                                        params, update)
    return new_params, {"m": new_m, "v": new_v, "step": t}

# Training loop
def L_pde(p, X_r):  return pde_residual_loss(p, X_r)
def L_bnd(p, X_b, U_b, X_ic, U_ic):  return bc_loss(p, X_b, U_b) + ic_loss(p, X_ic, U_ic)

state = init_multiadam(params, n_groups=2)

@jax.jit
def step(params, state, X_r, X_b, U_b, X_ic, U_ic):
    return multiadam_update(
        params, state,
        [lambda p, *a: L_pde(p, a[0]),
         lambda p, *a: L_bnd(p, a[1], a[2], a[3], a[4])],
        (X_r, X_b, U_b, X_ic, U_ic),
    )

for it in range(max_iter):
    params, state = step(params, state, X_r, X_b, U_b, X_ic, U_ic)
```

For coupled systems (e.g. Navier-Stokes with momentum + continuity), supply a list of `n=3` or more losses.

## Results
On Poisson with domains scaled from 8x8 to 1x1 (a 4096x PDE-loss change), MultiAdam keeps relative L2 at 2-4% while vanilla Adam degrades from 2.6% to 70%, LRA to 17%, NTK to 6%. The empirical per-group second-moment ratio matches the theoretically predicted scale-invariant weight `t^{2k}`. Also improves Helmholtz (nonlinear) and Burgers benchmarks by 1-2 orders of magnitude over baselines.
