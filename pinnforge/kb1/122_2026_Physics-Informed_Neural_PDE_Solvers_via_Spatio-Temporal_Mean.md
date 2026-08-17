---
slot: 122
title: "Physics-Informed Neural PDE Solvers via Spatio-Temporal MeanFlow"
authors: [Hanru Bai, Yuncheng Zhou, Difan Zou]
year: 2026
venue: arXiv:2605.08915
gitrepo: ""
---

## TL;DR
Adapt MeanFlow (a single-step continuous-time integrator from generative modeling) to PDE solving: parameterize the average integral `m` of the PDE right-hand side over a finite spatio-temporal path, then derive identities from `l·m = u(x,t)−u(ξ,τ)` and use them as physics-informed losses, decoupled into a Temporal-MF and a Spatial-MF term to soften the Hessian's spatio-temporal cross-coupling.

## Problem
PINNs use pointwise residuals (no integral perspective); neural operators (FNO, DeepONet) pre-discretize the time grid and behave like fixed step-size integrators. Neither matches the continuous-integral nature of PDEs `u(x,t) = u(x,τ) + ∫_τ^t f[u]ds`. Naive joint space-time MeanFlow has an ill-conditioned Hessian when temporal and spatial scales differ greatly.

## Method

### A. Spatio-temporal MeanFlow identity
Define straight path `p(s) = (ξ + s(x−ξ), τ + s(t−τ))` with length `l = sqrt(‖x−ξ‖² + (t−τ)²)`. Define
$$l\cdot m[u(\xi,\tau),\xi,\tau,x,t,a] = u(x,t) - u(\xi,\tau).$$
Differentiating w.r.t. `τ` and `ξ` yields the constraint set with `K = I + l ∂m/∂u`, `l_t = t−τ`, `l_s = ‖x−ξ‖`:
$$l_t\,m = lKf + \gamma l^2 \tfrac{\partial m}{\partial\tau},\qquad lK\nabla^\top u(\xi,\tau)=m(x-\xi)^\top - l^2\tfrac{\partial m}{\partial \xi}$$
Higher-order spatial derivatives (e.g. `∆u`) get analogous identities, supplied to substitute the gradients inside `f`.

### B. Decoupled losses (the key practical trick)
$$L_{\text{T-MF}} = \bigl\|m - Kf - \gamma l_t \tfrac{\partial m}{\partial \tau}\bigr\|_2^2,\quad L_{\text{S-MF}} = \bigl\|l_s K\nabla^\top u - m(x-\xi)^\top + l_s^2 \tfrac{\partial m}{\partial \xi}\bigr\|_2^2$$
Removes spatio-temporal cross-derivative residuals, making the Hessian nearly block-diagonal. Total loss `L = L_data + λ_t L_{T-MF} + λ_s L_{S-MF}`.

### C. Steady-state branch (γ=0)
Drop time; analytically extend boundary `u_0(x)` into interior, predict `m_θ` so `u(x) = u_0(x) + m_θ`. Only `L_data + λ_s L_{S-MF}`.

### D. Training and inference
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MeanFlowNet(nn.Module):
    """Predicts m given (u(x,tau), tau, x, t, a)."""
    hidden: int = 128
    depth: int = 4
    u_dim: int = 1
    @nn.compact
    def __call__(self, u_tau, tau, x, t, a):
        h = jnp.concatenate([u_tau, tau, x, t, a], axis=-1)
        for _ in range(self.depth):
            h = nn.gelu(nn.Dense(self.hidden)(h))
        return nn.Dense(self.u_dim)(h)

def m_apply(params, u_tau, tau, xi, t, a):
    return net.apply(params, u_tau, tau, xi, t, a)

def st_meanflow_losses(params, u_tau, xi, tau, x, t, a, pde_rhs):
    m = m_apply(params, u_tau, tau, xi, t, a)
    u_pred = u_tau + (t - tau) * m
    f_val  = pde_rhs(u_pred, x, t, a)
    # K = 1 + l_t * ∂m/∂u
    dm_du = jax.vmap(jax.jacrev(lambda u: m_apply(params, u, tau[0], xi[0], t[0], a[0])))(u_tau)
    K = 1.0 + (t - tau) * dm_du.reshape(m.shape)
    # ∂m/∂τ
    dm_dtau = jax.vmap(jax.jacrev(
        lambda tt: m_apply(params, u_tau[0], tt, xi[0], t[0], a[0])))(tau).reshape(m.shape)
    L_T = jnp.mean((m - K * f_val - (t - tau) * dm_dtau) ** 2)
    # ∂m/∂ξ
    dm_dxi = jax.vmap(jax.jacrev(
        lambda zz: m_apply(params, u_tau[0], tau[0], zz, t[0], a[0])))(xi).reshape(xi.shape)
    ls = (x - xi)
    grad_u = jax.vmap(jax.jacrev(
        lambda zz: u_tau[0] + (t[0] - tau[0]) *
                   m_apply(params, u_tau[0], tau[0], zz, t[0], a[0])))(xi).reshape(xi.shape)
    L_S = jnp.mean((ls * K * grad_u - m * (x - xi) + ls ** 2 * dm_dxi) ** 2)
    return L_T, L_S, u_pred

net = MeanFlowNet()

@jax.jit
def step(params, opt_state, u_tau, xi, tau, x, t, a, y, lam_t, lam_s):
    def loss_fn(p):
        L_T, L_S, u_pred = st_meanflow_losses(p, u_tau, xi, tau, x, t, a, pde_rhs)
        L_data = jnp.mean((u_pred - y) ** 2)
        return L_data + lam_t * L_T + lam_s * L_S
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss

# Multi-step inference: u_{k+1} = u_k + Δt * m_θ(u_k, t_k, x, t_{k+1}, a)
```

Hyperparameters: Adam, lr `5.5e-4` (Burgers), `2e-4` (NS), `3e-4` (steady-state). `λ_t = λ_s = 0.01`. Network: 4-layer MLP backbone with GELU, hidden 128.

## Results
Relative L2 (x10^-2) — Burgers 0.26, NS 0.19, Darcy 0.25, Poisson 0.18, HYCOM 23.43 — beats FNO/PINO/CFO/DiffusionPDE; inference 0.24 ms/sample at 0.012 GFLOPs (sub-FNO). Zero-shot resolution generalization (train s=128, test s=256/512 with no accuracy loss) and OOD initial-condition shifts both attributed to the spatial-MF loss. Multi-step inference best at 2-4 steps.
