---
slot: 043
title: "PhyCRNet: Physics-informed Convolutional-Recurrent Network for Solving Spatiotemporal PDEs"
authors: [Pu Ren, Chengping Rao, Yang Liu, Jian-Xun Wang, Hao Sun]
year: 2021
venue: Computer Methods in Applied Mechanics and Engineering (arXiv:2106.14103)
gitrepo: "https://github.com/isds-neu/PhyCRNet"
---

## TL;DR
Instead of representing `u(x,t)` with a fully-connected MLP, PhyCRNet uses an encoder–ConvLSTM–decoder over the discretised spatial grid and steps it autoregressively in time with a forward-Euler residual connection. PDE derivatives are computed by fixed high-order finite-difference convolution kernels and the loss is the discretised PDE residual; I/BCs are hard-encoded via padding so no IC/BC penalty terms are needed.

## Problem
MLP-based PINNs scale poorly for spatiotemporal PDEs: many collocation points, soft IC/BC penalties that fight the residual, and slow convergence for sharp fronts. Pure CNN autoregressive surrogates (AR-DenseED) accumulate errors.

## Method
**Architecture.** Each time step performs:
1. Encoder: 3 conv layers (channels 8→32→128, kernel 4×4, stride 2, ReLU, periodic padding).
2. ConvLSTM cell on the latent grid (128 hidden, kernel 3×3, stride 1):
$$ \begin{aligned}
i_t &= \sigma(W_i*[X_t,h_{t-1}]+b_i),\quad f_t = \sigma(W_f*[X_t,h_{t-1}]+b_f)\\
\tilde{C}_t &= \tanh(W_c*[X_t,h_{t-1}]+b_c),\quad C_t = f_t\odot C_{t-1}+i_t\odot \tilde{C}_t\\
o_t &= \sigma(W_o*[X_t,h_{t-1}]+b_o),\quad h_t = o_t\odot \tanh(C_t)
\end{aligned} $$
3. Decoder: PixelShuffle upsampling (factor 8) + scaling conv (5×5).
4. Global residual: `u_{i+1} = u_i + δt · NN(u_i; θ)` (forward Euler), so IC is encoded by `u_0`.

**Hard BCs** via padding: Dirichlet → fixed-value padding; Neumann → ghost nodes from internal field; periodic → circular padding (applied twice for 4th-order stencils).

**Differentiation by fixed FD filters** (no autograd needed):
$$ K_t = \tfrac{1}{2\delta t}[-1,0,1],\quad
K_s = \tfrac{1}{12(\delta x)^2}\begin{bmatrix}0&0&-1&0&0\\0&0&16&0&0\\-1&16&-60&16&-1\\0&0&16&0&0\\0&0&-1&0&0\end{bmatrix} $$

**Loss** (PDE residual only, summed over all grid points and time steps):
$$ \mathcal{L}(\theta) = \sum_{i,j,k} \big\| u_t^\theta + \mathcal{F}[u^\theta,\nabla u^\theta,\nabla^2 u^\theta,\dots;\lambda]\big\|_2^2 $$

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

def circular_pad(x, pad):
    return jnp.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='wrap')

class ConvLSTMCell(nn.Module):
    hid_ch: int
    k:      int = 3

    @nn.compact
    def __call__(self, x, h, c):
        z   = jnp.concatenate([x, h], axis=-1)
        z   = circular_pad(z, self.k // 2)
        z   = nn.Conv(features=4 * self.hid_ch, kernel_size=(self.k, self.k),
                      padding='VALID')(z)
        i, f, g, o = jnp.split(z, 4, axis=-1)
        c_new = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h_new = jax.nn.sigmoid(o) * jnp.tanh(c_new)
        return h_new, c_new

class PhyCRNet(nn.Module):
    in_ch:  int = 2
    hid_ch: int = 128
    r:      int = 8

    @nn.compact
    def __call__(self, u, h, c, dt):
        z = u
        for feat in (8, 32, self.hid_ch):
            z = circular_pad(z, 1)
            z = nn.Conv(feat, (4, 4), strides=(2, 2), padding='VALID')(z)
            z = nn.relu(z)
        h, c  = ConvLSTMCell(self.hid_ch)(z, h, c)
        up    = nn.ConvTranspose(self.hid_ch // (self.r * self.r),
                                 (self.r, self.r), strides=(self.r, self.r))(h)
        delta = nn.Conv(self.in_ch, (5, 5), padding='SAME')(up)
        return u + dt * delta, h, c

# fixed FD kernels (constants, not trained)
def laplacian(u, dx):
    k = jnp.array([[0,0,-1,0,0],[0,0,16,0,0],[-1,16,-60,16,-1],
                   [0,0,16,0,0],[0,0,-1,0,0]], dtype=u.dtype)
    k = k.reshape(5, 5, 1, 1) / (12 * dx * dx)
    k = jnp.tile(k, (1, 1, u.shape[-1], 1))
    return jax.lax.conv_general_dilated(
        circular_pad(u, 2), k,
        window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=u.shape[-1])

def rollout_loss(params, u0, T, dt, dx):
    B, H, W, C = u0.shape
    h = jnp.zeros((B, H // 8, W // 8, 128))
    c = jnp.zeros_like(h)
    u = u0
    total = 0.0
    for _ in range(T):
        u_new, h, c = model.apply(params, u, h, c, dt)
        u_t = (u_new - u) / dt
        res = u_t + (u * grad_x(u, dx) + v * grad_y(u, dx)) - nu * laplacian(u, dx)
        total = total + jnp.mean(res ** 2)
        u = u_new
    return total

model = PhyCRNet()
opt   = optax.adam(1e-3)
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, u0):
    grads = jax.grad(rollout_loss)(params, u0, T, dt, dx)
    upd, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, upd), opt_state
```

Hyperparameters: spatial grid 128×128; Adam, 10 000 epochs over `T/3 – 2T/3`; weight normalisation (no batch-norm). PhyCRNet-s periodically skips the encoder every `T` steps to save compute.

## Results
On 2-D Burgers, λ-ω and FitzHugh-Nagumo reaction-diffusion systems, PhyCRNet beats vanilla PINN and AR-DenseED in accuracy and shows much better extrapolation beyond the training time window, with strong generalisation to unseen ICs. PhyCRNet-s trades a small accuracy hit for additional speed.
<!-- input was pymupdf-fallback plain text but content was clear enough -->
