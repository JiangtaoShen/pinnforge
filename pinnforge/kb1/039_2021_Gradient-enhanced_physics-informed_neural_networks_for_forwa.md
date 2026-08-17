---
slot: 039
title: "Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems"
authors: [J. Yu, Lu Lu, Xuhui Meng, G. Karniadakis]
year: 2021
venue: Computer Methods in Applied Mechanics and Engineering (arXiv:2111.02801)
gitrepo: ""
---

## TL;DR
Since a PDE residual `f(x)=0` everywhere implies `∇f(x)=0` everywhere, gPINN augments the standard PINN loss with the squared norm of the residual's spatial gradient. This redundant constraint dramatically improves accuracy of both `u` and `∂u/∂x` with fewer collocation points, and combines naturally with residual-based adaptive refinement (RAR).

## Problem
Vanilla PINNs converge slowly and require many residual points to drive the loss low; predicted derivatives are usually much less accurate than the value field, which hurts inverse problems and quantities that depend on `∇u`.

## Method
Let the PDE residual be
$$ f(x) = \mathcal{N}\!\left[u; \nabla u, \nabla^2 u, \dots; \lambda\right] = 0,\; x\in\Omega $$
The gPINN loss adds one term per spatial dimension:
$$ \mathcal{L} = w_f \mathcal{L}_f + w_b \mathcal{L}_b + w_i \mathcal{L}_i + \sum_{i=1}^{d} w_{g_i}\,\mathcal{L}_{g_i} $$
$$ \mathcal{L}_{g_i}(\theta;T_{g_i}) = \frac{1}{|T_{g_i}|}\sum_{x\in T_{g_i}} \left|\frac{\partial f}{\partial x_i}\right|^2 $$
For 1-D Poisson `-u_xx = f(x)` the extra term is `(u_xxx - f'(x))^2`. The gradient is computed by a second autograd pass through the residual. Reuse `T_f` as `T_{g_i}`. Tune `w_g`: there is an optimum around `0.01–0.1`; values ≥ 1 can hurt.

Combine with RAR (Algorithm 1): train for `K` iterations, evaluate `|f|` on a large random pool, append `m` worst points to `T`, repeat.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    hidden: int = 20
    depth:  int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x)

model = MLP()

def u_fn(params, x):                          # scalar output per point
    return model.apply(params, x)

def residual_at(params, x):                   # 1-D Poisson example
    u    = lambda xx: u_fn(params, xx).sum()
    u_x  = jax.grad(u)
    u_xx = jax.grad(lambda xx: u_x(xx).sum())
    return -u_xx(x) - source(x)               # PDE-specific

def gpinn_loss(params, x_f, x_b, u_b, w_g=0.01):
    f = residual_at(params, x_f)
    L_f = jnp.mean(f ** 2)
    # one extra grad pass per dimension
    df = jax.jacrev(lambda xx: residual_at(params, xx).sum())(x_f)  # (N, d)
    L_g = jnp.sum(jnp.mean(df ** 2, axis=0))
    L_b = jnp.mean((u_fn(params, x_b) - u_b) ** 2)
    return L_f + L_b + w_g * L_g

opt = optax.adam(1e-3)
params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, d_in)))
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, x_f, x_b, u_b):
    grads = jax.grad(gpinn_loss)(params, x_f, x_b, u_b, 0.01)
    upd, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, upd), opt_state

for k in range(20_000):
    params, opt_state = step(params, opt_state, x_f, x_b, u_b)

# RAR loop
for cycle in range(n_rar):
    pool = sample_pool(N=10_000)
    r    = jnp.abs(residual_at(params, pool))
    idx  = jnp.argsort(r.squeeze())[-m:]
    x_f  = jnp.concatenate([x_f, pool[idx]], axis=0)
    # retrain K more steps...
```

Recommended hyper-parameters (Table 1 of the paper):
- 4 layers × 20 units, tanh, Adam lr=1e-3, 10–100 k iterations for simple problems
- Adam + L-BFGS for inverse problems
- gradient weight `w_g ≈ 0.01–0.1`
- enforce Dirichlet BC hard, e.g. `û(x) = x(π-x) N(x) + x`

## Results
On 1-D Poisson with 20 collocation points, gPINN (`w=0.01`) cuts the L2 error of `u` by ~10× and of `du/dx` by ~100× versus PINN. Adds Diffusion-Reaction, Burgers, wave, Brinkman-Forchheimer (inverse), and diffusion-reaction inverse: gPINN+RAR consistently outperforms PINN+RAR for solutions with steep gradients. Sensitivity to `w_g`: too large (≥1) is worse than baseline PINN.
