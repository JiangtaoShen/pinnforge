---
slot: 047
title: "Residual-based adaptivity for two-phase flow simulation in porous media using Physics-informed Neural Networks"
authors: [John Hanna, Jose V. Aguado, Sebastien Comas-Cardona, Ramzi Askri, Domenico Borzacchiello]
year: 2021
venue: Computer Methods in Applied Mechanics and Engineering (arXiv:2109.14290)
gitrepo: ""
---

## TL;DR
Standard RAR (residual-based adaptive refinement) over-clusters collocation points at sharp moving fronts. The paper instead samples new points from a probability density built from the (log) residual field, giving a smoother, more uniform reinforcement. The scheme is applied per-equation in a coupled two-phase Darcy flow PINN, and also to the IC/BC training points, yielding much more accurate front predictions than fixed-grid PINN or RAR at the same compute cost.

## Problem
Two-phase Darcy flow has a moving sharp front in the fraction function `c(x,t)`; classical fixed-grid PINNs miss it and RAR adds many points in a tiny region, hurting generalisation. Coupled PDEs share collocation points even when their residual hotspots differ.

## Method
**Governing equations.** Three coupled PDEs (Darcy, incompressibility, advection):
$$ f_1 = c_t + v\cdot\nabla c = 0,\quad f_2 = v + \tfrac{1}{\mu(c)}K\nabla p = 0,\quad f_3 = \nabla\!\cdot v = 0,\quad \mu(c)=c\mu_2+(1-c)\mu_1 $$
Three separate MLPs (5 layers × 20 tanh, sigmoid output for `c, p`; linear for `v`) approximate `c, p, v`.

**Loss.** Sum of `λ_i · (residual MSE on its own collocation set)` for `f_1, f_2, f_3` plus penalties for IC/BC sets `v·n, c-c_b, p-p_b`. All `λ_i = 1`.

**Adaptive sampling (per residual `r`)** — sample new collocation points from a density built from the log-residual:
$$ p_r(X) = \frac{\max(\log|r(X)/\varepsilon|, 0)}{\int_\Omega \max(\log|r(X)/\varepsilon|, 0)\,dX} $$
where the integral is estimated by Monte-Carlo over a dense pool. Each PDE residual gets its own pool and density, so the three sets `T_{f_1}, T_{f_2}, T_{f_3}` diverge — `T_{f_1}` clusters near the moving front; the others stay roughly uniform. IC/BC training points are enriched the same way using their residuals.

**Algorithm.** Outer loop of `M` adaptation rounds; each round (i) computes residuals on the pool, (ii) builds the per-PDE densities `p_i`, (iii) draws new points and adds them to the training set, (iv) trains for `n` more iterations.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from jaxopt import LBFGS

class MLP(nn.Module):
    d_out: int
    final: str = "linear"
    hidden: int = 20
    depth:  int = 5
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.hidden)(x))
        x = nn.Dense(self.d_out)(x)
        return jax.nn.sigmoid(x) if self.final == "sigmoid" else x

net_c = MLP(d_out=1, final="sigmoid")
net_p = MLP(d_out=1, final="sigmoid")
net_v = MLP(d_out=1)

def residuals(params, xt):
    def c_of(z): return net_c.apply(params['c'], z)
    def p_of(z): return net_p.apply(params['p'], z)
    def v_of(z): return net_v.apply(params['v'], z)
    c, p, v = c_of(xt), p_of(xt), v_of(xt)
    cx = jax.vmap(jax.grad(lambda z: c_of(z[None]).sum()))(xt)[:, 0:1]
    ct = jax.vmap(jax.grad(lambda z: c_of(z[None]).sum()))(xt)[:, 1:2]
    px = jax.vmap(jax.grad(lambda z: p_of(z[None]).sum()))(xt)[:, 0:1]
    vx = jax.vmap(jax.grad(lambda z: v_of(z[None]).sum()))(xt)[:, 0:1]
    mu = c * mu2 + (1 - c) * mu1
    f1 = ct + v * cx
    f2 = v + (k / mu) * px
    f3 = vx
    return f1, f2, f3

def sample_by_density(key, pool, residual, eps=1e-3, n_new=200):
    w   = jnp.clip(jnp.log(jnp.abs(residual) / eps), a_min=0.0).squeeze()
    w   = jnp.where(w.sum() == 0, jnp.ones_like(w), w)
    w   = w / w.sum()
    idx = jax.random.choice(key, pool.shape[0], (n_new,), replace=False, p=w)
    return pool[idx]

opt = optax.adam(1e-3)
opt_state = opt.init(params)
key = jax.random.PRNGKey(0)

# 1) initial Adam phase
@jax.jit
def adam_step(params, opt_state, T_f1):
    def total(p):
        f1, f2, f3 = residuals(p, T_f1)
        return (jnp.mean(f1**2) + jnp.mean(f2**2) + jnp.mean(f3**2)
                + bc_ic_losses(p))
    g = jax.grad(total)(params)
    upd, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state

for _ in range(1000):
    params, opt_state = adam_step(params, opt_state, T_f1)

# 2) BFGS + adaptive enrichment loop
for m in range(M):
    key, k1, k2, k3, kp = jax.random.split(key, 5)
    pool = jax.random.uniform(kp, (5000, 2))
    f1, f2, f3 = residuals(params, pool)
    T_f1 = jnp.concatenate([T_f1, sample_by_density(k1, pool, f1)])
    T_f2 = jnp.concatenate([T_f2, sample_by_density(k2, pool, f2)])
    T_f3 = jnp.concatenate([T_f3, sample_by_density(k3, pool, f3)])
    def total(p):
        f1, _, _ = residuals(p, T_f1); _, f2, _ = residuals(p, T_f2); _, _, f3 = residuals(p, T_f3)
        return (jnp.mean(f1**2) + jnp.mean(f2**2) + jnp.mean(f3**2)
                + bc_ic_losses(p))
    solver  = LBFGS(fun=total, maxiter=50)
    params, _ = solver.run(params)
```

Recommended hyperparameters: 5 layers × 20 tanh per network; Adam 1e-3 for 1000 iterations then BFGS; `ε` ≈ 10⁻³ to control spread; start with 40×40 grid, enrich every 50 BFGS iterations until tolerance; equal `λ_i = 1`.

## Results
1-D and 2-D injection problems. The proposed density-based adaptivity matches the analytical front position better than fixed-grid PINN or PINN+RAR, while taking similar wall-clock time (~200 s). Training–test loss gap shrinks dramatically with the new scheme, indicating reduced overfitting; RAR shows only marginal improvement over the fixed grid because its points concentrate too narrowly.
