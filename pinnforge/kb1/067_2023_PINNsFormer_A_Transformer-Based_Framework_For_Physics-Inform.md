---
slot: 67
title: "PINNsFormer: A Transformer-Based Framework for Physics-Informed Neural Networks"
authors: [Zhiyuan Zhao, Xueying Ding, B. Aditya Prakash]
year: 2023
venue: ICLR 2024 (arXiv:2307.11833)
gitrepo: "https://github.com/AdityaLab/pinnsformer"
---

## TL;DR
Turn each `(x, t)` collocation point into a **pseudo-sequence** `{(x, t), (x, t+dt), ..., (x, t+(k-1)dt)}` and feed it through an encoder-decoder Transformer with a new **Wavelet activation** `w1 sin(x) + w2 cos(x)`. The model outputs the solution at all `k` time-stamps; the PINN loss is averaged over the sequence (residual + BC on all steps, IC only on the first step). The self-attention captures temporal causality that MLP-PINNs lack and mitigates classic failure modes (convection-50, 1-D reaction).

## Problem
Vanilla MLP PINNs operate point-to-point and cannot model temporal dependency, so the IC information fails to propagate globally — they collapse to overly-smooth or trivial solutions on convection with `beta=30..50`, 1-D reaction, 1-D wave, and turbulent NS.

## Method

### A. Pseudo-sequence generator
For input `(x, t)`, produce a length-`k` sequence by stepping forward in time:
$$
[x, t] \mapsto \{[x, t],\,[x, t+\Delta t],\,\ldots,\,[x, t+(k-1)\Delta t]\}
$$
Typical `k = 5`, `dt` small (e.g. `1e-3..1e-2`).

### B. Architecture (Spatio-Temporal Mixer + Encoder-Decoder)
1. Linear "Spatio-Temporal Mixer": map each `R^d` sequence element to `R^D` (`D ~ 32-64`).
2. **Encoder** stack (`L_e ~ 1-2` blocks): self-attention + feedforward, GELU/Wavelet activations.
3. **Decoder** stack (`L_d ~ 1-2` blocks): encoder-decoder attention only (no self-attention; the encoded sequence serves as both query and key/value), feedforward.
4. Output MLP head produces `u_hat in R^{k x 1}`.

### C. Wavelet activation
$$
\mathrm{Wavelet}(x) = \omega_1 \sin(x) + \omega_2 \cos(x), \quad \omega_1, \omega_2 \text{ learnable}
$$
A two-hidden-layer wavelet network is a universal approximator via the Fourier integral (the paper's Prop. 1).

### D. Sequence-aware PINN loss
Residual + BC are imposed at all `k` time-stamps; IC only at `j=0`:
$$
\mathcal L_\text{res} = \frac{1}{k N_\text{res}}\sum_{i=1}^{N_\text{res}}\sum_{j=0}^{k-1}\|\mathcal D[\hat u(x_i, t_i + j\Delta t)] - f(x_i, t_i + j\Delta t)\|^2
$$
$$
\mathcal L_\text{bc} = \frac{1}{k N_\text{bc}}\sum_i\sum_j\|\mathcal B[\hat u(x_i, t_i + j\Delta t)] - g\|^2,\quad
\mathcal L_\text{ic} = \frac{1}{N_\text{ic}}\sum_i\|\mathcal I[\hat u(x_i, 0)] - h(x_i)\|^2
$$
$$
\mathcal L = \lambda_r\mathcal L_\text{res} + \lambda_b\mathcal L_\text{bc} + \lambda_i\mathcal L_\text{ic}\quad(\lambda_\cdot = 1)
$$
Auto-diff: per-sequence derivatives are computed independently per stamp (each `(x, t+j*dt)` is a fresh JAX input).

```python
import jax, jax.numpy as jnp
import flax.linen as nn

class Wavelet(nn.Module):
    @nn.compact
    def __call__(self, x):
        w1 = self.param("w1", nn.initializers.ones, (1,))
        w2 = self.param("w2", nn.initializers.ones, (1,))
        return w1 * jnp.sin(x) + w2 * jnp.cos(x)

class TransformerBlock(nn.Module):
    d_model: int; n_heads: int; self_attn: bool = True
    @nn.compact
    def __call__(self, x, memo=None):
        attn = nn.MultiHeadDotProductAttention(num_heads=self.n_heads)
        if self.self_attn:
            x = x + attn(x, x)
        else:
            x = x + attn(x, memo)
        x = nn.LayerNorm()(x)
        h = nn.Dense(4 * self.d_model)(x)
        h = nn.tanh(h)
        h = nn.Dense(self.d_model)(h)
        return nn.LayerNorm()(x + h)

class PINNsFormer(nn.Module):
    d_in: int = 2; d_model: int = 32; n_heads: int = 2
    k: int = 5; dt: float = 1e-3; L_e: int = 1; L_d: int = 1

    def make_pseudo(self, x, t):
        offsets = jnp.arange(self.k).reshape(1, self.k, 1) * self.dt
        T = t[:, None, :] + offsets                          # [B, k, 1]
        X = jnp.broadcast_to(x[:, None, :], (x.shape[0], self.k, x.shape[1]))
        return jnp.concatenate([X, T], axis=-1)              # [B, k, d_in]

    @nn.compact
    def __call__(self, x, t):
        seq = self.make_pseudo(x, t)
        h   = nn.Dense(self.d_model)(seq)
        h   = Wavelet()(h)
        h   = nn.Dense(self.d_model)(h)
        memo = h
        for _ in range(self.L_e):
            memo = TransformerBlock(self.d_model, self.n_heads, self_attn=True)(memo)
        dec = h
        for _ in range(self.L_d):
            dec = TransformerBlock(self.d_model, self.n_heads, self_attn=False)(dec, memo)
        head = nn.Dense(self.d_model)(dec)
        head = Wavelet()(head)
        return nn.Dense(1)(head)                              # [B, k, 1]

def pinnsformer_loss(params, apply_fn, X, T, fX_b, gX_b, hX_ic):
    def u_of(x_in, t_in):
        return apply_fn(params, x_in, t_in)                   # [B, k, 1]
    U = u_of(X, T)
    res = jax.vmap(lambda x, t: pde_op(params, apply_fn, x, t))(X, T) - fX_b
    L_res = jnp.mean(res**2)
    L_bc  = jnp.mean((B_op(U) - gX_b)**2)
    L_ic  = jnp.mean((I_op(U[:, 0:1]) - hX_ic)**2)
    return L_res + L_bc + L_ic
```

Optimizer: L-BFGS (Strong-Wolfe via `jaxopt`), 1000 iterations; `k = 5`, `dt = 1e-3`; param budget matched to baselines (~10k-50k). At test time take only the first sequence element `U[:, 0]`.

## Results
On convection (`beta=30`), 1-D reaction, 1-D wave, and 2-D Navier-Stokes (final time), PINNsFormer reaches relative RMSE `~0.03` while PINNs, QRes, and FLS get stuck at `~0.8`. Loss-landscape Lipschitz drops from 776 (PINN) to 33 (PINNsFormer). Compatible with NTK weighting (further 2-3x gain on 1-D wave). Cost: ~3x compute, ~2x memory vs MLP-PINN.
