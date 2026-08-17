---
slot: 62
title: "A unified scalable framework for causal sweeping strategies for PINNs and their temporal decompositions"
authors: [Michael Penwarden, Ameya D. Jagtap, Shandian Zhe, George Em Karniadakis, Robert M. Kirby]
year: 2023
venue: Journal of Computational Physics (arXiv:2302.14227)
gitrepo: "https://github.com/mpenwarden/dtPINN"
---

## TL;DR
Unifies time-marching, XPINNs, bc-PINNs, causal weighting, and adaptive time-sampling under one taxonomy (soft/hard causality on time-slab/sampling scale). Introduces two new methods: **stacked-decomposition** (a "causal XPINN" that progressively activates time sub-networks as the front reaches them) and **window-sweeping** (sweep a small active collocation window in time, combining hard + soft causality with transfer-learning warm starts).

## Problem
For long-time or stiff forward PDEs (long convection, Allen-Cahn, KdV, KS) standard continuous-time PINNs and XPINNs converge to the zero-solution, no-propagation, or wrong-propagation failure modes. Time-marching avoids this but is purely sequential; XPINNs parallelize but break information flow at interfaces; causal weights (Wang 2022) work but recompute residuals over the whole domain every step. A scalable hybrid is missing.

## Method

### A. Stacked-decomposition (causal XPINN)
Split `[0, T]` into `n` time-slabs, one sub-network per slab. Each step, only `d_S` consecutive sub-networks are *active* (trainable). The window slides forward: when slab `k`'s residual drops below tolerance, slab `k+d_S` becomes active and is **warm-started by transfer learning** from slab `k+d_S-1` (copy weights). Interfaces use C^p continuity (for first-order PDEs, solution continuity `MSE(u_k(T_k, x) - u_{k+1}(T_k, x))`) — not the XPINN `u_avg` form which the paper shows performs worse.

`d_S = 1` -> time-marching; `d_S = n` with warmup -> "causal XPINN"; intermediate gives middle ground that parallelizes well.

### B. Window-sweeping collocation algorithm
Maintain only `N_r^win` active collocation points inside a moving time window `[t_l, t_r]` rather than the whole domain. Push the window forward when the residual mean inside is small enough. Combine with the causal-weight loss (Wang 2022) **inside** the window:
$$
\mathcal L_r(\theta) = \frac{1}{N_t}\sum_{i: t_i \in [t_l, t_r]} \exp\!\Big(-\varepsilon \sum_{k<i, t_k \in [t_l, t_r]} \mathcal L_r(t_k, \theta)\Big)\,\mathcal L_r(t_i, \theta)
$$
Annealing for `eps`: increase by 10x when min weight exceeds threshold; window advance criterion: `mean(L_r over window) < tau_advance`.

Total loss for each subnet: `MSE_u (IC/BC) + MSE_r (residual on active points) + MSE_interface (C0 continuity)`.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    width: int = 100
    depth: int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.width)(x))
        return nn.Dense(1)(x)

def make_subnet(key, dummy_x):
    net = MLP()
    params = net.init(key, dummy_x)
    return net, params

def stacked_step(active, params_list, opt_states, optimizer,
                 X_r_per_slab, X_b, U_b, X_ic, U_ic, X_iface):
    """One synchronous gradient step over all currently active slabs."""
    def slab_loss(p_curr, p_prev_stop, k):
        X = X_r_per_slab[k]
        r = jax.vmap(lambda x: pde_residual(p_curr, mlp_apply, x))(X)
        L_r = jnp.mean(r**2)
        if k == 0:
            u_ic = jax.vmap(mlp_apply, in_axes=(None, 0))(p_curr, X_ic)
            L_ic = jnp.mean((u_ic - U_ic)**2)
        else:
            u_prev = jax.lax.stop_gradient(
                jax.vmap(mlp_apply, in_axes=(None, 0))(p_prev_stop, X_iface[k]))
            u_curr = jax.vmap(mlp_apply, in_axes=(None, 0))(p_curr, X_iface[k])
            L_ic   = jnp.mean((u_prev - u_curr)**2)
        u_bc = jax.vmap(mlp_apply, in_axes=(None, 0))(p_curr, X_b[k])
        L_bc = jnp.mean((u_bc - U_b[k])**2)
        return L_r + 100.0 * L_ic + 100.0 * L_bc

    new_params, new_states = list(params_list), list(opt_states)
    for k in range(active + 1):
        p_prev = params_list[k - 1] if k > 0 else params_list[k]
        grads  = jax.grad(slab_loss)(params_list[k], p_prev, k)
        upd, new_states[k] = optimizer.update(grads, opt_states[k], params_list[k])
        new_params[k]      = optax.apply_updates(params_list[k], upd)
    return new_params, new_states

def maybe_activate_next(active, params_list, residual_metric, tau):
    if residual_metric < tau and active < len(params_list) - 1:
        params_list[active + 1] = jax.tree_util.tree_map(jnp.copy, params_list[active])
        active += 1
    return active, params_list
```

Hyper-params: `n_slabs = 4..20`; per-slab MLP 4 layers x 100 tanh; Adam(1e-3) + L-BFGS finishing; `tau_advance ~ 1e-4..1e-6`; causal-weight `eps` annealed `0.01 -> 100`.

## Results
On long-time convection (T=5), Allen-Cahn, KS, and KdV — problems where vanilla PINN/XPINN converge to wrong/zero solutions — stacked-decomposition + window-sweeping reaches reference relative L2 below 1e-3 with 2-10x lower per-iteration cost than full-domain causal-weight PINN, because residuals are computed only on the active window.
