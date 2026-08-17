---
slot: 61
title: "A practical PINN framework for multi-scale problems with multi-magnitude loss terms"
authors: [Yong Wang, Yanzhong Yao, Jiawei Guo, Zhiming Gao]
year: 2023
venue: Journal of Computational Physics (arXiv:2308.06672)
gitrepo: "https://github.com/wangyong1301108/MMPINN"
---

## TL;DR
For multi-scale PDEs the supervised and residual loss terms can differ by 6+ orders of magnitude; minimizing their sum traps the optimizer in a bad local minimum where the BC/IC is violated. **MMPINN** flattens magnitudes by taking fractional powers `L_s^{1/m} + L_r^{1/n}` (with optional grouping across sub-domains), and pairs this with an **integrated network (INN)** that applies multiple Fourier-feature mappings, each fed into an improved-MLP gated block. The two together drop relative L2 by orders of magnitude.

## Problem
When `L_r >> L_s` (or vice versa), gradient descent first reduces the dominant term, often *increasing* the small one, and gets stuck before the IC/BC is fit. Standard adaptive weighting (LR-annealing, NTK, SA-PINN) only helps in moderate regimes. Additionally, high-frequency multi-scale solutions suffer from MLP spectral bias.

## Method

### A. Magnitude-flattening regularization
Replace the standard loss `w_s L_s + w_r L_r` with
$$
\widetilde{\mathcal L}(\theta) = w_s\,\mathcal L_s(\theta)^{1/m} + w_r\,\mathcal L_r(\theta)^{1/n},\qquad m,n > 0
$$
Pick `m, n` so the two reshaped terms have comparable order. E.g. if `L_r ~ 1e9`, `L_s ~ 1e3`, use `n = 3, m = 1` -> both ~`1e3`. Optionally use a **multi-level schedule**: train with `(m=1, n=3)` -> save `theta_1`; restart L-BFGS with `(m=1, n=2)` -> `theta_2`; final restart with `(m=1, n=1)` to recover the standard PINN loss with a good warm start. **Grouping** version: split the spatial/temporal domain into K sub-regions, give each its own residual term and exponent — handles within-residual magnitude imbalance.

### B. Integrated Neural Network (INN) backbone
For each of `N` scales `i = 1..N`, sample frequency matrix `B^{(i)}` with entries i.i.d. `Normal(0, sigma_i^2)`, then build a Fourier feature embedding
$$
F_i(x) = [\sin(B^{(i)} x),\, \cos(B^{(i)} x)]
$$
Two gating transforms per scale, fed to a gated-residual MLP (Wang et al. 2021 improved FC):
$$
U_i = \phi(W_u^i F_i(x) + b_u^i),\quad V_i = \phi(W_v^i F_i(x) + b_v^i)
$$
$$
Z_l^i = \phi(W_l H_{l-1}^i + b_l),\quad H_l^i = (1 - Z_l^i)\odot U_i + Z_l^i\odot V_i
$$
Final output: `u(x) = W_L * concat(H_{L-1}^1, ..., H_{L-1}^N) + b_L`. Typical `N = 2` scales with `sigma_1 = 1, sigma_2 = 25`; 3 hidden layers, 100 tanh units per layer.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class FourierFeature(nn.Module):
    hidden: int
    sigma: float

    @nn.compact
    def __call__(self, x):
        B = self.param("B", nn.initializers.normal(stddev=self.sigma),
                       (x.shape[-1], self.hidden))
        # Treat B as a fixed sampled matrix (paper freezes B)
        B = jax.lax.stop_gradient(B)
        proj = x @ B
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

class INNBlock(nn.Module):
    hidden: int
    depth: int
    sigma: float

    @nn.compact
    def __call__(self, x):
        f = FourierFeature(self.hidden // 2, self.sigma)(x)     # -> 2*(hidden//2) = hidden
        U = nn.tanh(nn.Dense(self.hidden)(f))
        V = nn.tanh(nn.Dense(self.hidden)(f))
        H = f
        for _ in range(self.depth):
            Z = nn.tanh(nn.Dense(self.hidden)(H))
            H = (1.0 - Z) * U + Z * V
        return H

class INN(nn.Module):
    hidden: int = 100
    depth:  int = 3
    sigmas: tuple = (1.0, 25.0)

    @nn.compact
    def __call__(self, x):
        feats = [INNBlock(self.hidden, self.depth, s)(x) for s in self.sigmas]
        return nn.Dense(1)(jnp.concatenate(feats, axis=-1))

def mmpinn_loss(params, apply_fn, X_r, X_s, U_s, m=1, n=3, w_s=1.0, w_r=1.0):
    r   = jax.vmap(lambda x: pde_residual(params, apply_fn, x))(X_r)
    L_r = jnp.mean(r**2)
    L_s = jnp.mean((jax.vmap(apply_fn, in_axes=(None, 0))(params, X_s) - U_s)**2)
    return w_s * L_s**(1.0 / m) + w_r * L_r**(1.0 / n)

# Multi-level: train with n=3, then n=2, then n=1 (standard)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)
for n_lvl in [3, 2, 1]:
    @jax.jit
    def step(params, opt_state, X_r, X_s, U_s):
        grads = jax.grad(lambda p: mmpinn_loss(p, apply_fn, X_r, X_s, U_s, 1, n_lvl))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state
    for _ in range(2000):
        params, opt_state = step(params, opt_state, X_r, X_s, U_s)
# Optionally end each level with jaxopt L-BFGS finishing.
```

Recommend Adam (2000 iters) then L-BFGS for each level. `tanh` activations; Xavier init.

## Results
On a large-gradient heat equation, a multi-frequency Poisson, and a strong-source heat conduction problem, MMPINN-INN reaches relative L2 of `1e-3..1e-5`, vs `1e-1..1e0` for vanilla PINN (even with `1e6` Adam iters), and beats LR-annealing, NTK-PINN, SA-PINN, and Fourier-only baselines.
