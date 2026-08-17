---
slot: 5
title: "Modeling the Dynamics of PDE Systems with Physics-Constrained Deep Auto-Regressive Networks"
authors: [N. Geneva, N. Zabaras]
year: 2019
venue: "Journal of Computational Physics (arXiv:1906.05747)"
gitrepo: "https://github.com/cics-nd/ar-pde-cnn"
---

## TL;DR
Replace the MLP+autograd PINN by a *convolutional* encoder-decoder (AR-DenseED) trained to act as a one-step time-integrator on a uniform grid. The "physics-informed" loss is the residual of a *finite-difference* discretisation of the PDE (e.g. Crank-Nicolson), not autograd; the trained net auto-regressively rolls out long time series from any initial condition. A Bayesian SWAG variant adds UQ.

## Problem
Standard MLP PINNs must be retrained when the initial condition changes; autograd-based spatial derivatives are expensive on high-resolution 2-D/3-D dynamical PDEs (Kuramoto-Sivashinsky chaos, multi-shock Burgers); fully-connected nets converge slowly on structured-grid problems.

## Method
Discretise `Omega` on a uniform grid; state at step `n` is `u_n in R^{d0 x D1 x D2}`. Use a CNN `f(chi_n; w)` mapping the last `k+1` snapshots to the next snapshot:
$$
u_{n+1} = f(\chi_{n+1}; w),\quad \chi_{n+1} = (u_n, u_{n-1},\dots,u_{n-k})
$$
Architecture: encoding conv blocks reduce spatial dim, interleaved with DenseNet blocks (constant feature-map size), then decoding (transpose-conv) blocks restore resolution. Inputs are stacked snapshots along channel dim. `k=1-3` previous steps; deeper `k` slows training without accuracy gain.

Physics-constrained loss = residual of a discrete time-integrator `T_dt`:
$$
\mathcal{L}(w) = \sum_{j=1}^{M}\sum_{i=1}^{N}\bigl\| f(\chi_{ij}; w) - T_{\Delta t}\!\bigl(U_{ij}, F_{\Delta x}\bigr)\bigr\|_2^{\,2}
$$
where `T_dt` is e.g. Crank-Nicolson:
$$
T_{\Delta t}(u^n, u^{n+1}) : u^{n+1} = u^n + \tfrac{\Delta t}{2}\bigl(F_{\Delta x}(u^n) + F_{\Delta x}(u^{n+1})\bigr)
$$
Spatial derivatives in `F_dx` use central finite differences implemented as fixed conv kernels — so the loss is autograd-friendly w.r.t. `w` but uses FD (not autograd) for spatial ops.

Training is *data-free*: sample random initial conditions `u_0,i ~ p(u_0)` (e.g. GRF or random Fourier), roll the model forward N steps, accumulate the loss above, backprop.

Bayesian extension: SWAG (Stochastic Weight Averaging-Gaussian) approximates posterior over `w` by SGD trajectory mean + low-rank covariance; samples give predictive distribution.

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

# Fixed FD kernels used by F_dx (passed as constants, not parameters).
LAP_1D  = jnp.array([1., -2., 1.])[None, None, None, :] / dx**2
GRAD_1D = jnp.array([-0.5, 0., 0.5])[None, None, None, :] / dx

class AR_DenseED(nn.Module):
    in_ch: int
    out_ch: int
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(48, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = DenseBlock(growth=16, layers=4)(x)
        x = nn.Conv(96, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = DenseBlock(growth=16, layers=4)(x)
        x = nn.ConvTranspose(48, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.ConvTranspose(self.out_ch, (3, 3), strides=(2, 2), padding="SAME")(x)
        return x

def F_dx(u):                                          # e.g. Burgers RHS via FD
    u_x  = jax.lax.conv(u, GRAD_KERNEL, (1, 1), "SAME")
    u_xx = jax.lax.conv(u, LAP_KERNEL,  (1, 1), "SAME")
    return -u * u_x + nu * u_xx

def crank_nicolson(u_n, u_np1, dt):
    return u_n + 0.5 * dt * (F_dx(u_n) + F_dx(u_np1))

net = AR_DenseED(in_ch=k + 1, out_ch=1)
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, k + 1, H, W)))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

def loss_fn(params, u0_batch):
    chi = jnp.concatenate([u0_batch] * (k + 1), axis=1)
    L = 0.0
    for n in range(N_steps):
        u_pred = net.apply(params, chi)
        u_target = crank_nicolson(chi[:, -1:], u_pred, dt)   # implicit residual
        L = L + jnp.mean((u_pred - u_target) ** 2)
        chi = jnp.concatenate([u_pred, chi[:, :-1]], axis=1)
    return L

@jax.jit
def train_step(params, opt_state, u0_batch):
    grads = jax.grad(loss_fn)(params, u0_batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

Recommended: Adam `lr=1e-3` then decay; mini-batch of initial conditions `M = 32`; rollout `N = 10-40` during training; Crank-Nicolson for parabolic, RK4 for hyperbolic.

## Results
On Kuramoto-Sivashinsky (chaotic), 1-D Burgers with multiple shocks, and 2-D coupled Burgers, AR-DenseED matches spectral / high-order FV reference for energy spectra and shock locations while running 10-100x faster than the reference solver at inference. SWAG variant provides calibrated uncertainty bands that widen in chaotic regimes.
