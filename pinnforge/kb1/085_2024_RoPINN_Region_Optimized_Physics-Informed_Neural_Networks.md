---
slot: 85
title: "RoPINN: Region Optimized Physics-Informed Neural Networks"
authors: [Haixu Wu, Huakun Luo, Yuezhou Ma, Jianmin Wang, Mingsheng Long]
year: 2024
venue: "NeurIPS 2024 (arXiv:2405.14369)"
gitrepo: "https://github.com/thuml/RoPINN"
---

## TL;DR
Standard PINNs minimise a scatter-point loss `L(u, S) = (1/|S|) sum_x L(u, x)`. RoPINN replaces each point by a **neighbourhood region** `L_r^region(u, x) = (1/|Omega_r|) int_{Omega_r} L(u, x+xi) dxi`, approximated by Monte-Carlo sampling. This implicitly enforces high-order PDE derivative constraints and tightens the generalisation bound. A **trust-region calibration** based on gradient-variance keeps the sampling region inside a low-variance neighbourhood.

## Problem
Point-wise PINN losses cannot guarantee approximation on the continuous domain. Adding high-order derivative regularisation is costly; variational PINNs need test functions and quadrature. A drop-in objective replacement is needed that improves generalisation and high-order satisfaction without extra autograd passes.

## Method
Region objective:
$$ \mathcal{L}^{\text{region}}_r(u_\theta, S) = \frac{1}{|S|}\sum_{x\in S}\frac{1}{|\Omega_r|}\int_{\Omega_r} \mathcal{L}(u_\theta, x+\xi)\,d\xi $$
Approximate by sampling `xi ~ U[0, sigma r]^(d+1)` *once per iteration* per point: `S' = {x_i + xi_i}`, then use the standard PINN loss on `S'`. No extra gradient calls.

**Trust-region calibration**: keep a buffer `g` of the last `T_0` per-step gradients, compute the per-parameter standard deviation `sigma(g)`, and set the next-step sampling scale `sigma_{t+1} = ||sigma(g)||`. This bounds the gradient estimation variance by limiting `Omega_r` to a region where the loss has roughly constant gradient.

Theoretical results: under L-Lipschitz, beta-smooth loss, region optimisation tightens the SGD generalisation bound by factor `(1 - |Omega_r|/|Omega|)`. Region optimisation also automatically generates high-order regularisation: differentiating the integral exposes derivatives of `L` on `x` of arbitrary order.

```python
import jax, jax.numpy as jnp
import optax
from collections import deque
from jax.flatten_util import ravel_pytree

def pinn_loss(params, S):                           # base loss on (possibly noisy) points
    x = S[:, :-1]; t = S[:, -1:]
    L_r  = jnp.mean(pde_residual(params, x, t)**2)
    L_ic = jnp.mean((net_apply(params, S_ic) - g_ic(S_ic[:, :-1]))**2)
    L_bc = bc_loss(params, S_bc)
    return L_r + lam_ic * L_ic + lam_bc * L_bc

class RoPINNTrainer:
    def __init__(self, params, base_loss_fn, r=0.01, T0=50, lr=1e-3):
        self.params = params
        self.base_loss_fn = base_loss_fn
        self.r, self.T0 = r, T0
        self.sigma = 1.0
        self.opt = optax.adam(lr)
        self.opt_state = self.opt.init(params)
        self.grad_hist = deque(maxlen=T0)

    def step(self, key, S):
        # 1) Monte-Carlo region sampling
        xi = jax.random.uniform(key, S.shape, minval=0.0, maxval=self.sigma * self.r)
        S_prime = S + xi
        loss, grads = jax.value_and_grad(self.base_loss_fn)(self.params, S_prime)
        # 2) Record per-parameter gradient as a flat vector
        g_flat, _ = ravel_pytree(grads)
        self.grad_hist.append(g_flat)
        # 3) Optax update
        updates, self.opt_state = self.opt.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        # 4) Trust-region calibration from gradient std
        if len(self.grad_hist) >= 2:
            G = jnp.stack(list(self.grad_hist))      # (T_window, P)
            self.sigma = float(jnp.linalg.norm(jnp.std(G, axis=0)))
        return float(loss)

# Usage
params  = ...                                       # any backbone (MLP / PINNsFormer / ModMLP)
trainer = RoPINNTrainer(params, pinn_loss, r=0.01, T0=50, lr=1e-3)
key = jax.random.PRNGKey(0)
for step in range(N):
    key, sk1, sk2 = jax.random.split(key, 3)
    S = sample_collocation(sk1, N=8192)
    trainer.step(sk2, S)
```

Hyperparameters: default region radius `r=0.01` (relative to domain extent); `T_0 = 10` past gradients buffered for variance; works with any backbone (canonical MLP, Modified-MLP, PINNsFormer, transformer). No extra autograd cost beyond sampling.

## Results
On 19 PDE benchmarks (Burgers, Navier-Stokes, Helmholtz, Wave, Heat, Convection-diffusion, etc.) and 5 different PINN backbones, RoPINN consistently reduces L2 error - often by 30-70% - over the corresponding point-optimised baseline, with identical wall-clock cost. Especially effective on high-order PDEs (wave) where the implicit derivative regularisation kicks in.
