---
slot: 84
title: "Respecting causality for training physics-informed neural networks"
authors: [Sifan Wang, Shyam Sankaran, Paris Perdikaris]
year: 2024
venue: "Computer Methods in Applied Mechanics and Engineering 421 (2024) 116813"
gitrepo: "https://github.com/PredictiveIntelligenceLab/CausalPINNs"
---

## TL;DR
Continuous-time PINNs trained with the standard mean residual loss are *biased toward fitting later times first*, violating physical causality. The fix is to weight each time slice's residual by `exp(-eps * sum_{k<i} L_r(t_k))`, so a slice is only "unlocked" when all earlier slices already have low residual. With this single modification, PINNs become the first method to simulate chaotic Lorenz, Kuramoto-Sivashinsky, and Navier-Stokes problems.

## Problem
The standard residual loss `L_r = (1/N_t) sum_i L_r(t_i, theta)` lets the optimiser minimise `L_r(t_i)` even when predictions at earlier `t_k < t_i` are wrong (errors then propagate). Empirically, the temporal NTK trace `C(t)` is monotone increasing in `t` at initialisation - confirming the bias.

## Method

### Causal weighting
For a temporal discretisation `0 = t_1 < ... < t_{N_t}`, define per-slice residual `L_r(t_i, theta) = (1/N_x) sum_j |du_theta/dt + N[u_theta]|^2 (t_i, x_j)`. Use weights
$$ w_i = \exp\Big(-\epsilon \sum_{k=1}^{i-1} \mathcal{L}_r(t_k, \theta)\Big),\quad \mathcal{L}_r(\theta) = \frac{1}{N_t}\sum_{i=1}^{N_t} w_i\,\mathcal{L}_r(t_i,\theta) $$
The weights `w_i` are **stop-gradient**: they shape the loss but are not differentiated. As earlier residuals fall, downstream weights `~ 1`, gradually exposing later times.

### Convergence criterion
Training is judged converged when `min_i w_i > delta` (e.g. `delta = 0.99`), meaning every time slice is being optimised. Tune `epsilon` adaptively: start small, increase whenever `min w_i > delta` (the authors use `epsilon in {1e-2, 1e-1, 1.0, 1e1, 1e2}`).

### Other practicalities
Use modified-MLP backbone (Wang et al. gated arch), Fourier feature embeddings for periodic BCs, NTK-style adaptive `lambda_ic, lambda_r`.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

def residual_AC(params, t, x):                      # Allen-Cahn
    inp = jnp.concatenate([t, x])
    u_of = lambda y: NET().apply(params, y).squeeze()
    grad_u = jax.grad(u_of)(inp)
    u_t = grad_u[0]
    u_xx = jax.hessian(u_of)(inp)[1, 1]
    u = u_of(inp)
    return u_t - 1e-4 * u_xx + 5 * u**3 - 5 * u

def per_slice_loss(params, t_i, x_grid):
    r = jax.vmap(lambda xj: residual_AC(params, t_i, xj))(x_grid)
    return jnp.mean(r**2)

def causal_loss(params, t_grid, x_grid, eps=1.0):
    L_r_per_t = jax.vmap(lambda ti: per_slice_loss(params, ti, x_grid))(t_grid)  # (N_t,)
    cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(L_r_per_t)[:-1]])
    w = jax.lax.stop_gradient(jnp.exp(-eps * cum))  # stop-gradient on weights
    return jnp.mean(w * L_r_per_t), w

def total_loss(params, t_grid, x_grid, x_ic, u0, lam_ic, eps):
    L_r, w = causal_loss(params, t_grid, x_grid, eps)
    L_ic   = ic_loss(params, x_ic, u0)
    return lam_ic * L_ic + L_r, w

optimizer = optax.adam(1e-3); opt_state = optimizer.init(params)
eps_schedule = [1e-2, 1e-1, 1.0]; eps_idx = 0; eps = eps_schedule[0]
delta = 0.99

@jax.jit
def step(params, opt_state, eps):
    (val, w), grads = jax.value_and_grad(total_loss, has_aux=True)(
        params, t_grid, x_grid, x_ic, u0, lam_ic, eps)
    upd, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, upd), opt_state, w

for it in range(N_iter):
    params, opt_state, w = step(params, opt_state, eps)
    if w.min() > delta and eps_idx < len(eps_schedule) - 1:
        eps_idx += 1; eps = eps_schedule[eps_idx]   # tighten causality
```

Hyperparameters: tanh MLP 5-10 x 128-512 (or modified-MLP); `N_t = 32-256` time slices, `N_x = 64-512` per slice; Adam `lr=1e-3` with exponential decay; `lambda_ic = 100-1e4`; long PDEs (Lorenz / KS) use time-marching of size `T_seg ~ 1-10` and repeat. Always periodic-BC via Fourier features when applicable.

## Results
Allen-Cahn rel-L2 `1.39e-4` with modified-MLP (vs `1.68e-2` for time-marching, `4.98e-1` for vanilla PINN). First PINN successes on the chaotic Lorenz system, Kuramoto-Sivashinsky in chaotic regime, and 2-D decaying-turbulence Navier-Stokes. The `min_i w_i` quantity is also a reliable a-posteriori convergence indicator.
