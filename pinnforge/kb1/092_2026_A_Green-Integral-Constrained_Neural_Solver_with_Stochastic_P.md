---
slot: 92
title: "A Green-Integral-Constrained Neural Solver with Stochastic Physics-Informed Regularization"
authors: [Mohammad Mahdi Abedi, David Pardo, Tariq Alkhalifah]
year: 2026
venue: arXiv:2604.21411
gitrepo: "https://github.com/mahdiabedi/Green-Integral-Neural-Solver-for-the-Helmholtz-Equation"
---

## TL;DR
For the frequency-domain acoustic Helmholtz scattering problem, replace the local PDE-residual loss with a global Lippmann-Schwinger Green-integral consistency loss (no second derivatives, no PML, FFT-accelerated). Optionally add a small PDE-residual penalty at non-uniformly importance-sampled collocation points concentrated where the scattering potential is large.

## Problem
For oscillatory Helmholtz at 10-20 Hz in heterogeneous seismic models (Marmousi, Overthrust, Otway), vanilla PINNs (i) require second-order derivatives that are costly and amplify spectral bias, (ii) admit the spurious total-field trivial solution `Us = -U0`, and (iii) need PML padding that 3-4x the domain. The PDE residual can be zero everywhere while the wavefield is non-physical.

## Method
Decompose the field `U = U0 + Us` where `U0` solves the background Helmholtz. The scattered field obeys the Lippmann-Schwinger relation
$$
U_s(x) = \omega^2 \int_\Omega G_0(x,y)\,\delta m(y)\,(U_0(y)+U_s(y))\,dy
$$
On a regular grid this is a discrete convolution with the Toeplitz kernel `G_0`, computed in `O(N log N)` via FFT:
$$
\widetilde U_s(y) = \mathcal F^{-1}\{\mathcal F(G_0)\cdot \mathcal F(D)\},\quad D(y)=\omega^2 \delta m(y)(U_0(y)+U_s(y))\,W
$$
Primary loss (no derivatives, no BCs):
$$
\mathcal L_{GI} = \frac1{N_y}\sum_j |U_s(y_j) - \widetilde U_s(y_j)|^2
$$
Optional hybrid with PDE residual at `N_x << N_y` importance-sampled points:
$$
\mathcal L = \mathcal L_{GI} + \lambda(t)\,\mathcal L_{PDE},\quad
\mathcal L_{PDE} = \frac1{N_x}\sum_i |\nabla^2 U_s(x_i)+\omega^2 m(x_i) U_s(x_i)+\omega^2\delta m(x_i)U_0(x_i)|^2
$$
with sigmoid schedule `lambda -> 0.01` and importance density `P(x) prop |delta m(x)|^alpha + epsilon` (alpha=1, eps=0.01 delta_m_max). Spatial coords are normalized by background wavelength.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class SineEncoder(nn.Module):
    in_dim: int
    K: int = 3
    @nn.compact
    def __call__(self, x):
        freqs = (2.0**jnp.arange(self.K)) * 2 * jnp.pi
        xf = x[..., None] * freqs                                 # (..., d, K)
        s, c = jnp.sin(xf), jnp.cos(xf)
        return jnp.concatenate([x, s.reshape(*x.shape[:-1], -1),
                                   c.reshape(*x.shape[:-1], -1)], axis=-1)

class GINet(nn.Module):                       # outputs (Re, Im) of U_s
    in_dim: int = 2; hidden: int = 128; depth: int = 5; K: int = 3
    @nn.compact
    def __call__(self, x):
        h = SineEncoder(self.in_dim, self.K)(x)
        h = jnp.sin(nn.Dense(self.hidden)(h))
        for _ in range(self.depth - 1):
            h = jnp.sin(nn.Dense(self.hidden)(h))
        re_im = nn.Dense(2)(h)
        return re_im[..., 0] + 1j * re_im[..., 1]

# precompute once
G0_pad = build_padded_green_kernel(grid, omega, v0)               # 2x size
G0_hat = jnp.fft.fftn(G0_pad)
U0_y   = compute_background_field(grid, omega, v0, src)
dm     = m_full - m0
W      = cell_area

def gi_recon(Us_y):
    D = (omega**2) * dm * (U0_y + Us_y) * W
    D_pad = pad_to_fft_size(D)
    return crop(jnp.fft.ifftn(G0_hat * jnp.fft.fftn(D_pad)))

net = GINet()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
sched  = optax.exponential_decay(1e-3, transition_steps=100_000,
                                 decay_rate=3.4e-4/1e-3)
optimizer = optax.adam(sched)
opt_state = optimizer.init(params)

def L_gi_fn(params, grid_pts):
    Us_y = net.apply(params, grid_pts)
    return jnp.mean(jnp.abs(Us_y - gi_recon(Us_y))**2)

@jax.jit
def step_gi(params, opt_state, grid_pts):
    g = jax.grad(L_gi_fn)(params, grid_pts)
    updates, opt_state = optimizer.update(g, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

@jax.jit
def step_hybrid(params, opt_state, grid_pts, x_pde, U0_pde, lam):
    def total(p):
        L_gi = L_gi_fn(p, grid_pts)
        L_pde = helmholtz_residual_complex(p, x_pde, m, dm, U0_pde, omega)
        return L_gi + lam * L_pde
    g = jax.grad(total)(params)
    updates, opt_state = optimizer.update(g, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

for epoch in range(100_000):
    if epoch < 5000:
        params, opt_state = step_gi(params, opt_state, grid_pts)
    else:
        idx = importance_sample(N_x, prob=jnp.abs(dm) + 1e-2)
        lam = 0.01 * jax.nn.sigmoid((epoch - 5000) / 5000.)
        params, opt_state = step_hybrid(params, opt_state, grid_pts,
                                         x_pool[idx], U0_pool[idx], lam)
```

Hyperparameters: 5 hidden layers x 128-150 neurons, sine activations + sinusoidal encoder (K=3), Adam lr 1e-3 -> 3.4e-4 exp decay, ~100k epochs. GI grid 120x170 to 532x330. PDE points: 2k-35k. Cell-averaged self-term for G0(0) (Appendix A) prevents log singularity.

## Results
On Marmousi 10 Hz: GI alone NMSE 0.079 in 8 min and 0.13 GB; 4x grid GI 0.025 in 11 min; hybrid GI+PDE 0.008 in 30 min. Baseline PINN+PML reaches 0.027 in 103 min and 3.7 GB (~10x slower, ~30x memory). Overthrust 10 Hz: GI 28 min vs PINN+PML 560 min (~20x). Otway 20 Hz (subwavelength layers): only GI variants are tractable.
