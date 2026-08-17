---
slot: 101
title: "AdamFLIP: Adaptive Momentum Feedback Linearization Optimization for Hard Constrained PINN Training"
authors: [Binghang Lu, Runyu Zhang, Changhong Mou, Na Li, Guang Lin]
year: 2026
venue: arXiv:2605.08408
gitrepo: ""
---

## TL;DR
Reformulate PINN training as an equality-constrained problem (minimize PDE residual subject to `L_ic = 0`, `L_bc = 0`; in inverse setting min `L_data` s.t. `L_phy = L_ic = L_bc = 0`) and solve it by feedback-linearization: compute the Lagrange multiplier in closed form so the constraint-violation dynamics become a stable linear contraction `y_{t+1} - y_t = -eta K y_t`. Then run Adam-style first/second-moment adaptation on the feedback-linearized Lagrangian gradient.

## Problem
Soft-penalty PINN losses `L = w_phy L_phy + w_ic L_ic + w_bc L_bc (+ w_data L_data)` are ill-conditioned and weight-sensitive; gradient pathologies arise. AL-PINN / trSQP-PINN partially fix this but rely on outer penalty-schedule loops, dual ascent, or trust-region QP solves; they're slow and still weight-sensitive.

## Method
Stack the constraint losses into a vector `h(theta) = [L_ic, L_bc]` (forward) or `[L_phy, L_ic, L_bc]` (inverse). The first-order KKT condition is `grad f + J_h^T lambda = 0`, `h = 0`. With `y_t = h(theta_t)`, requiring linear contraction `y_{t+1} - y_t = -eta K y_t` and using a first-order expansion of `y_{t+1}` yields the closed-form multiplier
$$
\lambda_t = -\big(J_h(\theta_t)J_h(\theta_t)^\top + \epsilon I\big)^{-1}\big(J_h(\theta_t)\nabla f(\theta_t) - K\,h(\theta_t)\big)
$$
with gain `0 prec K preceq (1/eta) I`. The feedback-linearized gradient is `g_t = grad f(theta_t) + J_h(theta_t)^T lambda_t`. Apply Adam's bias-corrected adaptive update on g_t:

```
m_t = beta1 m_{t-1} + (1-beta1) g_t
v_t = beta2 v_{t-1} + (1-beta2) g_t * g_t
m_hat = m_t / (1 - beta1^t),  v_hat = v_t / (1 - beta2^t)
theta_{t+1} = theta_t - eta * m_hat / (sqrt(v_hat) + delta)
```

```python
import jax, jax.numpy as jnp
import flax.linen as nn

class PINN(nn.Module):
    hidden: int = 64; depth: int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x).squeeze(-1)

net = PINN()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))

def flatten(p): return jnp.concatenate(
    [jnp.ravel(x) for x in jax.tree_util.tree_leaves(p)])

def unflatten_like(template, vec):
    leaves, tdef = jax.tree_util.tree_flatten(template)
    new = []; idx = 0
    for l in leaves:
        n = l.size; new.append(vec[idx:idx+n].reshape(l.shape)); idx += n
    return jax.tree_util.tree_unflatten(tdef, new)

def adamflip_step(params, loss_f, loss_h_fns, args_f, args_h_list,
                  state, K_gain=1.0, eta=1e-3, eps=1e-6, delta=1e-8,
                  beta1=0.9, beta2=0.999):
    # constraint values and Jacobian rows
    h_vals = jnp.stack([Lh(params, *a) for Lh, a in zip(loss_h_fns, args_h_list)])
    J_rows = [flatten(jax.grad(lambda p, a=a: Lh(p, *a))(params))
              for Lh, a in zip(loss_h_fns, args_h_list)]
    Jh = jnp.stack(J_rows)                                # [m, n_par]
    g_f = flatten(jax.grad(loss_f)(params, *args_f))      # [n_par]

    M   = Jh @ Jh.T + eps * jnp.eye(Jh.shape[0])
    rhs = Jh @ g_f - K_gain * h_vals
    lam = -jnp.linalg.solve(M, rhs)
    g_t = g_f + Jh.T @ lam

    state["t"]  = state["t"] + 1
    state["m"]  = beta1 * state["m"] + (1 - beta1) * g_t
    state["v"]  = beta2 * state["v"] + (1 - beta2) * g_t * g_t
    m_hat = state["m"] / (1 - beta1**state["t"])
    v_hat = state["v"] / (1 - beta2**state["t"])
    step  = eta * m_hat / (jnp.sqrt(v_hat) + delta)

    new_params = unflatten_like(params, flatten(params) - step)
    return new_params, state

n_par = flatten(params).size
state = {"t": 0,
         "m": jnp.zeros(n_par),
         "v": jnp.zeros(n_par)}

for it in range(max_iter):
    params, state = adamflip_step(
        params,
        loss_f      = pde_residual_mse,
        loss_h_fns  = [ic_mse, bc_mse],
        args_f      = (X_phy,),
        args_h_list = [(X_ic,), (X_bc,)],
        state=state, K_gain=1.0, eta=1e-3)
```

For inverse problems, treat `kappa` as extra trainable scalars, set `f = L_data`, constraints `h = [L_phy(theta, kappa), L_ic(theta), L_bc(theta)]`. The Jacobian rows `J_h` are computed by per-constraint `jax.grad`. The `(JJ^T + eps I)^{-1}` linear system is `m x m` where `m` is the number of constraint losses (small), so each step costs a few backward passes plus a tiny dense solve.

Hyperparameters: 4 hidden layers x 64 tanh, Adam (b1=0.9, b2=0.999), step eta=1e-3, gain K=I, damping eps=1e-6. Theorem 1 (metric-compatible variant, Algorithm 2) shows best-iterate KKT residual decays as `O(log T / sqrt(T))`.

## Results
Four benchmarks, forward and inverse: 1-D Burgers, 1-D time-fractional mixed diffusion-wave (TFMDWE), 2-D heat, 2-D Navier-Stokes Taylor-Green vortex. AdamFLIP has the lowest relative L2 on every problem. Headlines: Burgers fwd 3.12e-2 (vs 8.20e-2 AL-PINN); Burgers inv 5.74e-2 (vs ~4-5e-1 baselines), `kappa1` error 1.54e-2; TFMDWE fwd 2.85e-1 (vs 4.26e-1 FL-PINN); 2-D heat fwd 7.53e-3 vs Standard PINN 3.43e-2; 2-D Navier-Stokes velocity L2 ~1.1e-2 (3x better than next baseline). Wall-clock overhead vs Standard PINN <= 30%; trSQP-PINN is 2-5x slower.
