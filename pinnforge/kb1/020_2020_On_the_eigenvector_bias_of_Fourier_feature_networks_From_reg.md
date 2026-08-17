---
slot: 020
title: "On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks"
authors: [Sifan Wang, Hanwen Wang, Paris Perdikaris]
year: 2020
venue: "CMAME (arXiv:2012.10047)"
gitrepo: "https://github.com/PredictiveIntelligenceLab/MultiscalePINNs"
---

## TL;DR
Vanilla PINNs are biased toward learning low-frequency components of the solution because the dominant NTK eigenvectors are low-frequency. Wrap network inputs in one or several random Fourier-feature embeddings with different sigmas to push high-frequency content into the NTK's leading eigenvectors and recover multi-scale PDE solutions.

## Problem
For solutions with high-frequency or multi-scale content (e.g. u = sin(2 pi x) + 0.1 sin(50 pi x)), tanh-MLP PINNs cannot fit the high-frequency component even after 10^7 Adam steps. NTK analysis shows the leading eigenvectors of the limiting kernel concentrate around zero frequency, so gradient descent learns those first and stalls on higher frequencies.

## Method
A. Single-scale random Fourier feature embedding (Tancik et al. style):
$$
\gamma(x) = [\cos(2\pi B x);\ \sin(2\pi B x)],\quad B_{ij}\sim\mathcal{N}(0,\sigma^2),\ B\text{ fixed}
$$
sigma directly controls which frequency band the NTK favors.

B. Multi-scale Fourier embedding for PINNs. Apply M parallel embeddings with sigma_1,...,sigma_M (typical 1, 20, 50, 100). Each passes through the SAME MLP (shared weights), the M outputs are concatenated and projected by a final linear layer:
$$
\gamma^{(i)}(x) = [\cos(2\pi B^{(i)} x);\sin(2\pi B^{(i)} x)], \quad H^{(i)}_1 = \phi(W_1 \gamma^{(i)} + b_1)
$$
$$
f_\theta(x) = W_{L+1}\,[\,H^{(1)}_L;\ldots;H^{(M)}_L\,] + b_{L+1}
$$

C. Spatio-temporal variant: separate embeddings for x and t, merged by element-wise multiplication then a linear head — mimics Fourier spectral method u(x,t) = sum_k hat_u_k(t) e^{ikx}.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import math

class FourierEmbed(nn.Module):
    m: int
    sigma: float
    @nn.compact
    def __call__(self, x):
        # B is a frozen (non-trainable) buffer drawn from N(0, sigma^2)
        B = self.variable("buffers", "B",
                          lambda: jax.random.normal(self.make_rng("buffers"),
                                                    (self.m, x.shape[-1])) * self.sigma).value
        proj = 2.0 * math.pi * x @ B.T
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)

class Trunk(nn.Module):
    width: int = 100
    depth: int = 4
    @nn.compact
    def __call__(self, h):
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.width)(h))
        return h

class MultiScaleFFN(nn.Module):
    d_out: int
    sigmas: tuple
    m: int = 100
    width: int = 100
    depth: int = 4
    @nn.compact
    def __call__(self, x):
        trunk = Trunk(self.width, self.depth)  # one shared trunk module instance
        feats = [trunk(FourierEmbed(self.m, s, name=f"emb_{i}")(x))
                 for i, s in enumerate(self.sigmas)]
        return nn.Dense(self.d_out)(jnp.concatenate(feats, axis=-1))

class SpaceTimeFFN(nn.Module):
    d_out: int
    sigmas_x: tuple
    sigmas_t: tuple
    m: int = 100
    width: int = 100
    depth: int = 4
    @nn.compact
    def __call__(self, x, t):
        tx = Trunk(self.width, self.depth)
        tt = Trunk(self.width, self.depth)
        Hx = [tx(FourierEmbed(self.m, s, name=f"ex_{i}")(x))
              for i, s in enumerate(self.sigmas_x)]
        Ht = [tt(FourierEmbed(self.m, s, name=f"et_{i}")(t))
              for i, s in enumerate(self.sigmas_t)]
        merged = [hx * ht for hx in Hx for ht in Ht]
        return nn.Dense(self.d_out)(jnp.concatenate(merged, axis=-1))
```

Training is standard PINN: L = L_r + L_b with autograd PDE residuals (nested `jax.grad`). Use tanh, m=100 features per scale, 2-9 layers x 100-200 units (problem-dependent), `optax.adam(1e-3)`. B is fixed at init (not trained) — stored in the `buffers` collection so it is excluded from `params`.

## Results
On 1D Poisson with multi-scale solution, the multi-scale FFN drives error to ~1e-3 where vanilla PINN diverges. A reaction-diffusion (Gray-Scott) inverse problem that vanilla PINNs cannot solve is recovered with the architecture; the wave benchmark reaches ~1e-3 relative L2 only when this architecture is combined with the adaptive-weights training algorithm (architecture alone still fails). No added trainable parameters.
