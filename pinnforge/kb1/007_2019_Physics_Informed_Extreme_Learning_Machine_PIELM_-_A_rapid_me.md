---
slot: 7
title: "Physics Informed Extreme Learning Machine (PIELM) - A rapid method for the numerical solution of partial differential equations"
authors: [Vikas Dwivedi, Balaji Srinivasan]
year: 2019
venue: "Neurocomputing (arXiv:1907.03507)"
gitrepo: ""
---

## TL;DR
Replace the deep PINN by a single-hidden-layer network with *random fixed* input weights and biases (an Extreme Learning Machine). Only the output-layer weights `c` are trained, and they are obtained in one shot from a least-squares solve `H c = K` where rows of `H` encode the PDE residual + BC + IC at collocation points. No gradient descent, no autograd; orders of magnitude faster than PINN for linear PDEs.

## Problem
PINN training time is dominated by SGD/Adam over deep networks with autograd; small linear PDEs do not need that capacity. PIELM trades depth/expressivity for direct linear-algebra training that is exact (for linear PDEs) up to machine precision.

## Method
Single hidden layer with `N*` neurons, input `chi = [x, t, 1]^T`, tanh activation. Hidden weights `[m_k, n_k, b_k]` (one row per neuron) are sampled randomly from a uniform/Gaussian distribution and *frozen*. Output:
$$
f(x,t) = \sum_{k=1}^{N^*} c_k\,\phi(m_k x + n_k t + b_k)
$$
Because the input weights are known constants, derivatives are analytic:
$$
\frac{\partial^p f_k}{\partial x^p} = m_k^p\,\phi^{(p)}(z_k),\qquad \frac{\partial f_k}{\partial t} = n_k\,\phi'(z_k)
$$
For a linear PDE `L u = R`, with BC `u = B` on `dOmega` and IC `u(.,0) = F`, write residuals at `N_f` interior points, `N_bc` boundary points, `N_ic` IC points and demand they all vanish:
$$
\underbrace{\begin{pmatrix} L\,\phi(z) \\ \phi(z)\big|_{\partial\Omega} \\ \phi(z)\big|_{t=0}\end{pmatrix}}_{H \in \mathbb{R}^{(N_f+N_{bc}+N_{ic})\times N^*}}\,c \;=\;
\underbrace{\begin{pmatrix} R \\ B \\ F \end{pmatrix}}_{K}
$$
Solve `c = H^+ K` (Moore-Penrose pseudo-inverse). Pick `N* >= N_f + N_bc + N_ic` for guaranteed solvability.

DPIELM (Distributed PIELM): partition `Omega` into subdomains, train one PIELM per subdomain, enforce continuity of `f` and `df/dx` (or chosen derivatives) at interface points as additional rows of `H`. Handles sharp gradients that a single PIELM cannot represent.

JAX:
```python
import jax, jax.numpy as jnp

def pielm_solve(coords_int, coords_bc, coords_ic,
                L_op, B_fn, F_fn, R_fn,
                N_star=200, scale=2.0, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    d_in = coords_int.shape[1]                              # e.g. 2 for (x,t)
    W = (jax.random.uniform(k1, (N_star, d_in)) * 2 - 1) * scale  # random m,n
    b = (jax.random.uniform(k2, (N_star,)) * 2 - 1) * scale       # random bias
    phi = jnp.tanh

    def basis(coords):
        z = coords @ W.T + b                                # [N_pts, N*]
        return phi(z), z

    # H rows for PDE residual: L_op applies the operator analytically to each
    # basis function, returning a [N_int, N*] matrix.
    H_int = L_op(coords_int, W, b)
    H_bc, _ = basis(coords_bc)
    H_ic, _ = basis(coords_ic)
    H = jnp.concatenate([H_int, H_bc, H_ic], axis=0)
    K = jnp.concatenate([R_fn(coords_int),
                         B_fn(coords_bc),
                         F_fn(coords_ic)], axis=0)
    c, *_ = jnp.linalg.lstsq(H, K)                          # pseudo-inverse solve
    return W, b, c

@jax.jit
def predict(coords, W, b, c):
    return jnp.tanh(coords @ W.T + b) @ c
```

For the diffusion operator `L u = u_t - nu u_xx`, the row of `H` at interior point `(x_i, t_i)` is
`(n_k * phi'(z_ik) - nu * m_k^2 * phi''(z_ik))` for each neuron `k`.

Recommended (paper): `N* = 100-500`, tanh activation, uniform weights in `[-2, 2]`. For DPIELM use 4-16 subdomains, enforce `C^1` continuity.

## Results
On 1-D/2-D linear advection, diffusion, advection-diffusion in star-shaped and complex domains, PIELM matches or beats PINN accuracy (relative L2 ~ 1e-4 to 1e-6) and is 10-100x faster (seconds vs minutes). DPIELM resolves sharp-gradient problems where both PINN and single PIELM fail. Limitation: linear PDEs only; quasi-linear requires Picard/Newton outer loop.
