---
slot: 120
title: "PILIR: Physics-Informed Local Implicit Representation"
authors: [Jianfeng Li, Feng Wang, Ke Tang]
year: 2026
venue: arXiv:2605.00385
gitrepo: ""
---

## TL;DR
PILIR replaces the global MLP-coordinate-to-field map of standard PINNs with a (learnable feature grid) → (local generative neural decoder) → (physical head) pipeline. Each query point retrieves the 2^d enclosing vertex features, an MLP synthesises a continuous feature contribution per vertex from `(z_i, x − x_i)`, and these are spatially blended via cosine re-weighting to guarantee C^∞ continuity. The "generative" replacement of bilinear interpolation breaks the convex-hull constraint and resolves sub-grid high-frequency detail.

## Problem
PINNs with global MLPs exhibit strong spectral bias — high-frequency PDE components converge slowly or not at all (Helmholtz with k=10, multi-scale convection, Allen-Cahn). Grid-based PINNs (PIXEL, PIG) help but their fixed bilinear/Gaussian interpolation acts as a low-pass filter, smearing sub-grid structure unless grids are very fine (memory blowup in 3D).

## Method

### A. Discrete encoding
Partition `Ω⊂R^d` into a single-resolution grid; learnable feature tensor `Z ∈ R^{N_1×···×N_d×C}` with `C=4`-`8`. For query `x` normalised to `[0,1]^d`, anchor index `x_anchor = ⌊x·N⌋`; retrieve 2^d corner vectors `{z_i}`.

### B. Continuous feature synthesis (generative, not interpolated)
For each corner `v_i` with offset `Δx_i = x − x_i`, a shared decoder `f_{θ_s}: R^C × R^d → R^H` (small MLP, 2-3 hidden x 16) produces a contribution `h_i(x)=f_{θ_s}(z_i, Δx_i)`. Aggregate with cosine-reweighted volume weights for C^∞ continuity:
$$w'_i(x)=\tfrac12\bigl(1-\cos(\pi w_i(x))\bigr),\qquad h(x)=\sum_{i=1}^{2^d} w'_i(x)\,f_{\theta_s}(z_i,\Delta x_i)$$

### C. Physical decoding
Lightweight head `φ_ψ: R^H → R^{D_out}` (linear or 1-hidden MLP, tanh): `û(x)=φ_ψ(h(x))`.

### D. Loss
Standard PINN composite with equal weights `λ_r=λ_ic=λ_bc=1`.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax, itertools

class PILIR(nn.Module):
    d: int = 2
    N: int = 16
    C: int = 4
    H: int = 16
    hidden: int = 16
    depth: int = 2
    out_dim: int = 1

    @nn.compact
    def __call__(self, x):                                    # x: (B, d) in [0,1]^d
        Z = self.param("Z", lambda k: jax.random.uniform(
                       k, (self.N,) * self.d + (self.C,), minval=-0.1, maxval=0.1))
        N, d = self.N, self.d
        scaled = x * N
        anchor = jnp.clip(jnp.floor(scaled).astype(jnp.int32), 0, N - 2)
        frac = scaled - anchor.astype(scaled.dtype)
        # shared decoder f_s
        def f_s(zi, dx):
            h = jnp.concatenate([zi, dx], axis=-1)
            for _ in range(self.depth):
                h = jnp.tanh(nn.Dense(self.hidden)(h))
            return nn.Dense(self.H)(h)
        h_sum = 0.0
        for corner in itertools.product([0, 1], repeat=d):    # 2^d corners
            ct = jnp.array(corner)
            idx = anchor + ct                                  # (B, d)
            z = Z[tuple(idx[:, k] for k in range(d))]          # (B, C)
            x_v = idx.astype(scaled.dtype) / N
            dx = x - x_v
            w = jnp.ones(x.shape[0])
            for k in range(d):
                w_k = jnp.where(ct[k] == 0, 1 - frac[:, k], frac[:, k])
                w = w * w_k
            w = 0.5 * (1 - jnp.cos(jnp.pi * w))                # C^infty smoothing
            h_sum = h_sum + w[:, None] * f_s(z, dx)
        # physical head
        out = jnp.tanh(nn.Dense(self.hidden)(h_sum))
        return nn.Dense(self.out_dim)(out)

net = PILIR()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))

def total_loss(params, X_int, X_ic, X_bc, u_ic, u_bc, pde_residual):
    def u_fn(x):                                              # scalar input -> scalar
        return net.apply(params, x[None]).squeeze()
    res = jax.vmap(lambda x: pde_residual(u_fn, x))(X_int)
    L_r  = jnp.mean(res ** 2)
    L_ic = jnp.mean((jax.vmap(u_fn)(X_ic) - u_ic) ** 2)
    L_bc = jnp.mean((jax.vmap(u_fn)(X_bc) - u_bc) ** 2)
    return L_r + L_ic + L_bc

schedule = optax.cosine_decay_schedule(1e-2, 100_000, alpha=1e-4)
opt = optax.adam(schedule); opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, X_int, X_ic, X_bc, u_ic, u_bc):
    loss, grads = jax.value_and_grad(total_loss)(
        params, X_int, X_ic, X_bc, u_ic, u_bc, pde_residual)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss
```

Hyperparameters: grid `1×16×16` (2D) or `16×16×16` (3D / time), `C=4` (8 for reaction-diffusion), decoder 2-3 layers x 16 tanh, decoding head 1 hidden x 16 tanh. Optimiser Adam, lr cosine-annealed from 1e-2 (Helmholtz/NS) or 1e-3 (others) to 1e-6, 100k epochs. 10k PDE points + 1k IC/BC (2D); 5x in 3D.

## Results
Relative L2 (forward problems): Helmholtz-3D 5.04e-2 (PINN 2.47, PIG OOM); Helmholtz-2D 1.36e-2 (PINN 1.62); multi-scale convection 1.46e-1 (PINN 3.65e-1); Allen-Cahn 3.26e-2; reaction-diffusion 1.90e-2. On NS inverse problem recovers `λ_1=1.00, λ_2=1.06e-2` with finest vorticity and pressure-gradient details. Robust to coarse grids: PILIR-8x8 already beats PIXEL-16x16x16 on Allen-Cahn and Helmholtz.
