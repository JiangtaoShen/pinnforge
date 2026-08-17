---
slot: 68
title: "Physics-Informed Kernel Function Neural Networks for Solving Partial Differential Equations"
authors: [Zhuojia Fu, Wenzhi Xu, Shuainan Liu]
year: 2023
venue: Neural Networks (arXiv:2306.02606)
gitrepo: ""
---

## TL;DR
**PIKFNN** replaces the RBF activations of an RBF network with **physics-informed kernel functions** (PIKFs) — analytical functions, such as fundamental solutions or radial Trefftz/T-complete functions, that already satisfy the homogeneous governing PDE. The network is a single hidden layer of PIKFs centered at off-domain "source" nodes; only the linear output weights are trained, and **only boundary/initial data** appear in the loss — the PDE residual is identically zero by construction.

## Problem
PINNs are expensive for linear PDEs, high-wavenumber, unbounded-domain, or long-time problems because the residual loss requires interior collocation, repeated autodiff, and deep MLPs. Traditional RBF networks are cheap but lose accuracy on these same problems. Method of Fundamental Solutions (MFS) is fast but is ill-conditioned with poor source placement.

## Method
For a homogeneous PDE `L0 u = 0` on `Omega` with BC `B u = g` on `dOmega`:
1. Choose a PIKF `phi(x; s)` with `L0 phi(.; s) = 0` for any "source" point `s` outside `Omega` (a fictitious boundary). Examples: 2-D Laplace `phi = ln r`; 2-D Helmholtz `phi = H_0^{(1)}(k r)`; 2-D modified Helmholtz `phi = K_0(k r)`; convection-diffusion `phi = exp(v.r/(2D)) K_0(... r)`.
2. Distribute `M` source nodes `{s_j}` on a smooth fictitious surface enclosing (or enclosed by) `Omega`, and `N_B` boundary collocation nodes `{x_i^B}` on `dOmega`.
3. The trial solution is the single-layer expansion:
$$
u_{PIKFNN}(x) = \sum_{j=1}^{M} w_j\,\phi(x;\,s_j)
$$
which **identically satisfies the PDE everywhere in `Omega`**. Train only `w in R^M` to fit the BC:
$$
\mathcal L_{PIKFNN}(w) = \frac{1}{N_B}\sum_{i=1}^{N_B}\!\big(\mathcal B[u_{PIKFNN}](x_i^B) - g(x_i^B)\big)^2
$$
Linear least-squares -> closed-form `w = A^+ b` with `A_{ij} = B[phi(.; s_j)](x_i^B)`. Or train with Adam/L-M for nonlinear `B`.

4. **Nonhomogeneous** `L0 u = f`: pick a second operator `L1` with `L1 f = 0`; add `M'` PIKFs of `L1 L0` as a "particular solution" branch, doubling neurons. For `f = 0` chain length 1 suffices.

5. **Transient PDE**: choose space-time PIKFs `phi(x,t; s, tau)` of the time-dependent operator; sources placed in space-time. Loss adds an IC term.

```python
import jax, jax.numpy as jnp
import optax
import math

# PIKF kernels (all written in JAX; bessel functions via jax.scipy.special)
def phi_laplace_2d(x, sources):
    diff = x[:, None, :] - sources[None, :, :]
    r    = jnp.maximum(jnp.linalg.norm(diff, axis=-1), 1e-12)
    return jnp.log(r) / (2.0 * math.pi)

def phi_helmholtz_2d(x, sources, k):
    diff = x[:, None, :] - sources[None, :, :]
    r    = jnp.maximum(jnp.linalg.norm(diff, axis=-1), 1e-12)
    # Re part of H_0^{(1)}(kr) = J_0(kr)
    return jax.scipy.special.bessel_jn(k * r, v=0)

def phi_mod_helmholtz_2d(x, sources, k):
    diff = x[:, None, :] - sources[None, :, :]
    r    = jnp.maximum(jnp.linalg.norm(diff, axis=-1), 1e-12)
    # K_0(kr) — implement via scipy on host or jax custom; placeholder:
    return jax.scipy.special.k0e(k * r) * jnp.exp(-k * r) / (2.0 * math.pi)

def init_pikfnn(sources):
    return {"w": jnp.zeros((sources.shape[0], 1))}

def predict(params, x, sources, kernel_fn, **kw):
    return kernel_fn(x, sources, **kw) @ params["w"]          # [N, 1]

def train_homogeneous(sources, X_B, U_B, kernel_fn, mode="lstsq", **kw):
    A = kernel_fn(X_B, sources, **kw)                         # [N_B, M]
    if mode == "lstsq":
        w, *_ = jnp.linalg.lstsq(A, U_B, rcond=None)
        return {"w": w}
    params = init_pikfnn(sources)
    opt = optax.adam(1e-2); state = opt.init(params)
    @jax.jit
    def step(params, state):
        def loss(p):
            return jnp.mean((predict(p, X_B, sources, kernel_fn, **kw) - U_B)**2)
        g = jax.grad(loss)(params)
        u, state = opt.update(g, state, params)
        return optax.apply_updates(params, u), state
    for _ in range(5000):
        params, state = step(params, state)
    return params
```

Recommended: place sources on a smooth fictitious boundary 0.3-1.0 problem-length-scales offset from `dOmega`; `M ~ N_B`; train by Levenberg-Marquardt or direct least squares (closed form for Dirichlet BC). For Neumann or nonlinear BCs, switch to Adam/L-M.

## Results
On high-wavenumber Helmholtz (`k = 30` 2-D scattering), infinite-domain Laplace, nonhomogeneous modified Helmholtz, long-time transient heat conduction, spatial-fractional diffusion, an inverse EMG problem, and 3-D Laplace, PIKFNN matches analytical solutions to `1e-6..1e-10` relative L2 with `M < 200` parameters and seconds of training — orders of magnitude more accurate and faster than vanilla PINN, which needs deep nets, dense collocation, and minutes to hours.
