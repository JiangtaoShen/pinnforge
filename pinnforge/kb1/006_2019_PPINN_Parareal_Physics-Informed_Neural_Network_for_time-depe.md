---
slot: 6
title: "PPINN: Parareal Physics-Informed Neural Network for time-dependent PDEs"
authors: [Xuhui Meng, Zhen Li, Dongkun Zhang, George Em Karniadakis]
year: 2019
venue: "Computer Methods in Applied Mechanics and Engineering (arXiv:1909.10145)"
gitrepo: ""
---

## TL;DR
Apply the classical Parareal parallel-in-time algorithm to PINNs: a serial *coarse-grained* (CG) solver propagates a simplified PDE through the full long-time interval; many *fine* PINNs (one per time-subdomain) run in parallel and correct the CG initial-conditions iteratively. Yields multi-x speed-up versus one monolithic long-time PINN.

## Problem
A single PINN trained over a long time horizon `[0, T]` needs many residual points and a large network; loss landscape becomes intractable. Naively splitting time into subdomains does not work because each subdomain's IC is unknown.

## Method
Partition `[0, T]` into `N` equal subdomains `[t_i, t_{i+1}]`, `dT = T/N`. Two propagators:
- `G(u_i)`: cheap *coarse* solver (e.g. small PINN solving a simplified PDE — replace nonlinear coefficient by a constant, drop high-order terms — or a coarse FD solver), runs serially through all subdomains.
- `F(u_i)`: fine PINN trained on subdomain `i` with initial state `u_i`. All N fine PINNs run in *parallel*.

Parareal update at iteration `k`:
$$
u_{i+1}^{k+1} = G(u_{i+1}^{k}) + F(u_i^{k}) - G(u_i^{k}), \qquad i = 0,\dots,N-1
$$
The differences `F - G` correct the cheap prediction; `G` propagates the correction serially. Iterate `k = 0, 1, ...` until
$$
E_k = \frac{\sum_{i}\|u_i^{k+1} - u_i^{k}\|^2}{\sum_{i}\|u_i^{k+1}\|^2} < E_{tol}\;(\sim 1\%)
$$
Typically converges in `K = 2-4` iterations. Speed-up vs serial PINN:
$$
S \approx \frac{N \cdot \tau_f^1}{N\cdot\tau_c^0 + \tau_f^1 + N K \tau_c^k + (K-1)\tau_f^k}
$$

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class PINN(nn.Module):
    h: int = 20
    depth: int = 4
    @nn.compact
    def __call__(self, xt):
        for _ in range(self.depth):
            xt = jnp.tanh(nn.Dense(self.h)(xt))
        return nn.Dense(1)(xt)

def train_pinn(net, params, pde_fn, t0, t1, u_init, x_init, steps=5000, lr=1e-3):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(p, xt_r, x_ic):
        L_r  = jnp.mean(pde_fn(net, p, xt_r) ** 2)
        ic_in = jnp.concatenate([x_ic, jnp.full_like(x_ic, t0)], axis=1)
        L_ic = jnp.mean((net.apply(p, ic_in) - u_init) ** 2)
        return L_r + L_ic

    @jax.jit
    def step(p, s, xt_r, x_ic):
        g = jax.grad(loss_fn)(p, xt_r, x_ic)
        u, s = optimizer.update(g, s, p)
        return optax.apply_updates(p, u), s

    for _ in range(steps):
        xt_r = sample_residual(t0, t1)
        params, opt_state = step(params, opt_state, xt_r, x_init)
    return params

# Initialise interface states via serial CG
coarse = PINN(h=10, depth=2)
c_params = coarse.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
u = [u0]
for i in range(N):
    c_params = train_pinn(coarse, c_params, sPDE_residual, T[i], T[i+1], u[-1], x_grid, steps=1000)
    end = jnp.concatenate([x_grid, jnp.full_like(x_grid, T[i+1])], axis=1)
    u.append(jax.lax.stop_gradient(coarse.apply(c_params, end)))

# Parareal iterations — fine PINNs are independent and parallelisable (jax.pmap / vmap).
fine = PINN()
fine_params = [fine.init(jax.random.PRNGKey(i+1), jnp.zeros((1, 2))) for i in range(N)]
for k in range(K_max):
    F_vals = []
    for i in range(N):
        fine_params[i] = train_pinn(fine, fine_params[i], PDE_residual,
                                    T[i], T[i+1], u[i], x_grid, steps=2000)
        end = jnp.concatenate([x_grid, jnp.full_like(x_grid, T[i+1])], axis=1)
        F_vals.append(jax.lax.stop_gradient(fine.apply(fine_params[i], end)))
    G_vals_old = [coarse_eval(u[i], T[i], T[i+1]) for i in range(N)]
    u_new = [u0]
    for i in range(N):
        G_new = coarse_eval(u_new[-1], T[i], T[i+1])
        u_new.append(G_new + F_vals[i] - G_vals_old[i])
    if rel_error(u_new, u) < 1e-2: break
    u = u_new
```

Recommended: `N = 4-32` subdomains; CG = PINN with `h=10, depth=2` solving simplified linear PDE (or a fast FD solver); fine PINNs `h=20-40, depth=4-6`; tolerance 1%.

## Results
On 1-D Burgers and 2-D nonlinear diffusion-reaction, PPINN converges in 2-3 parareal iterations and yields wall-clock speed-ups roughly proportional to `N` (up to 24x in their tests) versus a monolithic long-time PINN, with comparable solution accuracy.
