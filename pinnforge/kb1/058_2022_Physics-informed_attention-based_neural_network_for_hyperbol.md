---
slot: 058
title: "Physics-informed attention-based neural network for hyperbolic partial differential equations: application to the Buckley-Leverett problem"
authors: [Ruben Rodriguez-Torrado, Pablo Ruiz, Luis Cueto-Felgueroso, Michael Cerny Green, Sebastien Matringe]
year: 2022
venue: Scientific Reports
gitrepo: ""
doi: 10.1038/s41598-022-11058-2
---

## TL;DR
Replace the plain MLP backbone of a PINN with an **encoder-decoder GRU + Bahdanau attention** that outputs the full spatial saturation profile at once. Hard-encode initial and boundary conditions in the architecture and discretize space on a fixed grid; the attention scores let the network locate shocks without adding artificial dissipation. Solves the non-convex-flux Buckley-Leverett equation for a continuum of mobility ratios `M`.

## Problem
Vanilla MLP-based PINNs cannot capture hyperbolic shocks in Buckley-Leverett (two-phase flow) for non-convex flux unless they add artificial viscosity, problem-specific physical constraints, or pre-clustered training points along the shock. The authors argue the bottleneck is the **architecture**, not the loss: a pointwise MLP cannot see the global spatial pattern needed to localize a discontinuity.

## Method
Inputs are `(t, M)`; output is a length-`N+1` vector of saturations on a fixed spatial grid `{x_0,...,x_N}`. Hard constraints in the architecture: prepend `u(t,M)_0 = 1` (Dirichlet BC at `x=0`) and the IC is enforced by setting `u(0,M)_i = 0`; a final sigmoid keeps outputs in `[0,1]`.

Backbone:
1. Encoder GRU sweeps `i = 1..N` over space, hidden state `h_0` from an MLP applied to `(t,M)`. Outputs `y_1,...,y_N` (latent code per spatial location).
2. Decoder GRU sweeps the same `N` positions, hidden state `d_0 = h_N`. Each block `g_i` outputs `u_i = g_i([u_{i-1}, c_i], d_{i-1})`.
3. **Attention** between decoder state `d_{i-1}` and every encoder output `y_j`:
$$
E_{i,j} = a(d_{i-1}, y_j),\qquad \alpha_{i,j} = \frac{\exp E_{i,j}}{\sum_{k} \exp E_{i,k}},\qquad c_i = \sum_{j=1}^{N} \alpha_{i,j}\, y_j
$$

PDE residual is the Buckley-Leverett equation `du/dt + df/dx = 0` with `f(u) = u^2 / (u^2 + (1-u)^2 / M)`. Time derivative `R1` is obtained either by central finite difference between adjacent time stamps **or** by `jax.grad`; spatial derivative `R2` is central finite difference between adjacent grid columns. Loss is the Frobenius norm `||R1 + R2||_F^2` averaged over the `M` minibatch — no IC/BC terms because they are baked in.

```python
import jax, jax.numpy as jnp
import flax.linen as nn

class PIANN(nn.Module):
    N: int
    hidden: int = 64

    @nn.compact
    def __call__(self, t, M, x_grid):           # t,M: [B,1]; x_grid: [N]
        h = nn.Dense(self.hidden)(jnp.concatenate([t, M], axis=-1))
        h = nn.tanh(h)
        h = nn.Dense(self.hidden)(h)
        gru_enc = nn.GRUCell(features=self.hidden)
        gru_dec = nn.GRUCell(features=self.hidden)
        # Encoder
        ys = []
        h_e = h
        for i in range(self.N):
            inp = jnp.broadcast_to(x_grid[i].reshape(1, 1), (t.shape[0], 1))
            h_e, _ = gru_enc(h_e, inp)
            ys.append(h_e)
        Y = jnp.stack(ys, axis=1)               # [B,N,hidden]
        # Decoder with Bahdanau attention
        d = h_e
        u_prev = jnp.zeros((t.shape[0], 1))
        us = [jnp.ones_like(u_prev)]            # BC: u(x=0) = 1
        Wd  = nn.Dense(self.hidden)
        Wy  = nn.Dense(self.hidden)
        v_a = nn.Dense(1)
        out = nn.Dense(1)
        for i in range(self.N):
            score = v_a(nn.tanh(Wd(d)[:, None, :] + Wy(Y)))      # [B,N,1]
            a = jax.nn.softmax(score, axis=1)
            c = jnp.sum(a * Y, axis=1)          # [B,hidden]
            d, _ = gru_dec(d, jnp.concatenate([u_prev, c], axis=-1))
            u_i = jax.nn.sigmoid(out(d))
            us.append(u_i)
            u_prev = u_i
        U = jnp.concatenate(us, axis=1)         # [B, N+1]
        return U * (t > 0)                      # IC: u(t=0) = 0 (except x=0)

def bl_residual(params, apply_fn, t, M, x_grid):
    def U_of_t(t_in):
        return apply_fn(params, t_in, M, x_grid)       # [B, N+1]
    U     = apply_fn(params, t, M, x_grid)
    dUdt  = jax.jacrev(U_of_t)(t)                       # per-row d/dt
    f     = U**2 / (U**2 + (1.0 - U)**2 / jnp.clip(M, 1e-6))
    dfdx  = (f[:, 2:] - f[:, :-2]) / (x_grid[2:] - x_grid[:-2])
    return dUdt[..., 1:-1] + dfdx
```

Training: Adam, lr `1e-3`, 200 epochs, grid `N=101, T=51`, `M in {2,4,...,100}`, batched over `(t, M)` pairs.

## Results
On non-convex-flux Buckley-Leverett, PIANN reproduces the analytical rarefaction + shock front for `M in [2, 100]` with cumulative residual `<1e-4` after a few epochs and no artificial diffusion. Generalizes to unseen `M` values inside the convex hull of the training set; vanilla PINNs (Fuks & Tchelepi) fail on the same problem without hand-crafted flux modifications.
