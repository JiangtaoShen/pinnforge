---
slot: 100
title: "A scaled TW-PINN: A physics-informed neural network for traveling wave solutions of reaction-diffusion equations with general coefficients"
authors: [Seungwan Han, Kwanghyuk Park, Jiaxi Gu, Jae-Hun Jung]
year: 2026
venue: arXiv:2603.15331
gitrepo: ""
---

## TL;DR
For traveling-wave solutions of `n`-D reaction-diffusion `d_t u = D Delta u + R(u)`, apply a scaling transformation `tau = rho t, xi = sqrt(rho/D) x` that normalizes both coefficients to 1, then use the traveling-wave coordinate `zeta = n . xi - c tau` to reduce the problem to a 1-D ODE. Train a single tiny PINN (one trainable wave-speed scalar `omega`, one hidden layer, one sigmoid output constraint) once on the scaled equation; reuse it for any `D, rho, n`.

## Problem
PINNs fail to capture sharp wave fronts when `rho/D` is large; existing wave-PINN reweights residuals but its predicted wave speed drifts when `rho >> 1`. The width of the transition layer scales as `sqrt(D/rho)`, so the same network has to resolve features 1000x narrower as `rho` grows; collocation density per unit wavelength collapses.

## Method
Scaling transformation removes both coefficients:
$$
\tau=\rho t,\;\;\xi=\sqrt{\rho/D}\,x,\;\;v(\xi,\tau)=u(x,t)\;\Rightarrow\;\partial_\tau v=\nabla_\xi^2 v + v^p(1-v^q)(v-a)^r
$$
Traveling-wave ansatz `v(xi, tau) = V(zeta)`, `zeta = n . xi - c tau`, makes the problem identically 1-D in any dimension and reduces it to `V'' + c V' + V^p(1 - V^q)(V - a)^r = 0`.

Network: input `(xi, tau)`, wave layer `zeta_hat = xi - omega tau` (omega is a learnable scalar), one hidden layer of N sigmoid neurons, output constraint
$$
\hat v(\xi,\tau) = \phi\Big(\sum_{i=1}^{N} c_i\sigma(a_i\hat\zeta + b_i)\Big),\quad
\phi(s) = v_- + (v_+ - v_-)\,\sigma(s)
$$
where `(v-, v+)` are the equilibrium states `(1, 0)` (Fisher/NWS/Zeldovich) or `(a, 1)` (bistable). The sigmoid output enforces monotone profile bounded inside the equilibria, killing spurious oscillations.

Loss (standard):
$$
\mathcal L = \frac1{N_{ICBC}}\sum |\hat v(\xi_i,\tau_i)-v_i|^2 + \frac1{N_r}\sum \big|\partial_\tau \hat v + \mathcal N[\hat v]\big|^2,\quad \mathcal N[v] = -\partial_{\xi\xi} - v^p(1-v^q)(v-a)^r
$$
Trainable parameters: `{a_i, b_i, c_i}` and `omega`. Note `omega` converges to the true wave speed `c`; monitoring it is a physical-convergence diagnostic. Use a *restricted* training domain `[-500, 500] x [0, 20]` instead of the original huge box `[-5000, 5000] x [0, 2000]` -- this concentrates collocation near the front and avoids spurious convergence.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

V_MINUS, V_PLUS = 1.0, 0.0          # Fisher/NWS/Zeldovich; use (a, 1) for bistable

class TWPINN(nn.Module):
    N: int = 64
    @nn.compact
    def __call__(self, xi, tau):
        omega = self.param("omega", nn.initializers.ones, ())
        zh    = (xi - omega * tau)[..., None]
        h     = jax.nn.sigmoid(nn.Dense(self.N)(zh))
        s     = nn.Dense(1)(h).squeeze(-1)
        return V_MINUS + (V_PLUS - V_MINUS) * jax.nn.sigmoid(s)

net = TWPINN(N=64)
params = net.init(jax.random.PRNGKey(0), jnp.zeros(()), jnp.zeros(()))

def v_at(params, xi, tau): return net.apply(params, xi, tau)

def residual(params, xi, tau, p, q, a, r):
    def v_scalar(xv, tv): return v_at(params, xv, tv)
    v    = jax.vmap(v_scalar)(xi, tau)
    v_t  = jax.vmap(jax.grad(v_scalar, argnums=1))(xi, tau)
    v_x  = jax.vmap(jax.grad(v_scalar, argnums=0))(xi, tau)
    v_xx = jax.vmap(jax.grad(jax.grad(v_scalar, argnums=0), argnums=0))(xi, tau)
    return v_t - v_xx - v**p * (1 - v**q) * (v - a)**r

def total(params, xi_r, tau_r, xi_b, tau_b, v_b, p, q, a, r):
    L_r    = jnp.mean(residual(params, xi_r, tau_r, p, q, a, r)**2)
    L_icbc = jnp.mean((jax.vmap(lambda xv, tv: v_at(params, xv, tv))(xi_b, tau_b) - v_b)**2)
    return L_r + L_icbc

sched     = optax.cosine_decay_schedule(1e-2, decay_steps=100_000)
optimizer = optax.adam(sched)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, xi_r, tau_r, xi_b, tau_b, v_b, p, q, a, r):
    g = jax.grad(total)(params, xi_r, tau_r, xi_b, tau_b, v_b, p, q, a, r)
    updates, opt_state = optimizer.update(g, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

# Sobol LHS on restricted box [-500,500] x [0,20]
from scipy.stats import qmc
LHS = qmc.Sobol(2, scramble=True).random(1024)
xi_r  = jnp.asarray(LHS[:, 0] * 1000 - 500)
tau_r = jnp.asarray(LHS[:, 1] * 20)
for it in range(100_000):
    params, opt_state = step(params, opt_state, xi_r, tau_r,
                              xi_b, tau_b, v_b, 1, 1, 0.0, 0)
```

Solution pipeline for arbitrary `(D, rho, n)`: scale `(t, x) -> (tau, xi)`, feed `(n, xi, tau)` to the wave layer (extended to multi-D as `zeta_hat = n . xi - omega tau`), invert `v_hat -> u_hat`.

Hyperparameters: 1 hidden layer with 64 sigmoid neurons, Adam lr 1e-2 with cosine annealing, 100k epochs, 1024 IC/BC + 1024 residual LHS points, restricted box `[-500, 500] x [0, 20]`.

## Results
For Fisher / NWS (q in {2,3,4}) / Zeldovich / bistable, |c - omega| ~ 1e-6 on the restricted domain (vs 1e-4 on the original box). Same trained solver, reused via the scaling pipeline, gives L2 errors of comparable magnitude across `rho in {1, 1e2, 1e4, 1e6}`. Outperforms wave-PINN at large rho (where wave-PINN's wave-front and speed are wrong). Universal-approximation theorem (Theorem 4.4) proves the architecture can approximate any traveling wave to arbitrary epsilon provided `omega = c`.
