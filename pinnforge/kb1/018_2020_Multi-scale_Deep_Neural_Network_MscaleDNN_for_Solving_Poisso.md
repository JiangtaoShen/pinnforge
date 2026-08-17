---
slot: 18
title: "Multi-scale Deep Neural Network (MscaleDNN) for Solving Poisson-Boltzmann Equation in Complex Domains"
authors: [Ziqi Liu, Wei Cai, Zhi-Qin John Xu]
year: 2020
venue: "Communications in Computational Physics (arXiv:2007.11207)"
gitrepo: ""
---

## TL;DR
Counter the F-Principle (spectral bias: PINNs learn low frequencies first) by splitting the first hidden layer into `N` groups, each of which sees the input pre-multiplied by a different *scale factor* `a_i` (e.g. `a_i = i` or `a_i = 2^{i-1}`). The high-frequency part of the target solution is thereby re-mapped to a low-frequency learning problem for its group. Pair with *compactly-supported* activations (sReLU or quadratic B-spline) for cleaner scale separation. Result: uniform fast convergence across multiple frequency bands - vital for Poisson-Boltzmann electrostatics on singular molecular surfaces.

## Problem
Standard MLP PINNs exhibit the F-Principle: gradient flow learns low spatial frequencies orders of magnitude faster than high frequencies; multi-scale PDE solutions (molecular electrostatics, oscillatory Helmholtz) are poorly resolved at high-frequency modes.

## Method
### A. Scaled-input first layer (MscaleDNN-1)
Partition first-layer neurons into `N` groups; group `i` receives input scaled by `a_i` (broadcast multiply, not separate weights). Effective forward pass:
$$
f_\theta(x) = W^{[L-1]}\sigma\!\bigl(\cdots W^{[1]}\sigma(W^{[0]}(K\odot x) + b^{[0]}) + b^{[1]}\cdots\bigr)
$$
where `K` is a fixed vector that broadcasts `a_i` to the `i`-th group of input rows of `W^[0]`:
$$
K = (\underbrace{a_1,...,a_1}_{m_1},\,\underbrace{a_2,...,a_2}_{m_2},\,\dots,\,\underbrace{a_N,...,a_N}_{m_N})
$$
The Hadamard `K odot x` shrinks high-frequency content in group `i` by factor `a_i`, mapping it to the "easy" low-frequency regime for that group's sub-network.

### B. MscaleDNN-2 (separate subnetworks)
Train `N` parallel small MLPs `f_{theta_i}(a_i x)` and sum: `f(x) = sum_i f_{theta_i}(a_i x)`. Slightly less parameter-efficient than MscaleDNN-1 but cleaner scale separation.

### C. Compact-support activations
Replace ReLU/tanh by:
- `sReLU(x) = ReLU(x) * ReLU(1 - x) = (x)_+ (1-x)_+` (triangular hat)
- Quadratic B-spline: `phi(x) = (x)_+^2 - 3(x-1)_+^2 + 3(x-2)_+^2 - (x-3)_+^2`

These have compact spatial support => Fourier transform is localised => each scale group spans a localised frequency band.

### D. Loss (Ritz energy)
For Poisson-Boltzmann `-div(eps grad u) + kappa^2 sinh(u) = rho`, minimise the Ritz functional discretised by Monte-Carlo on `Omega` plus boundary penalty:
$$
\mathcal{L} = \int_\Omega \bigl[\tfrac{\epsilon}{2}|\nabla u|^2 + \kappa^2(\cosh u - 1) - \rho u\bigr]dx + \beta\!\!\int_{\partial\Omega}(u - g)^2 d\sigma
$$

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

def sReLU(x):
    return jnp.maximum(x, 0) * jnp.maximum(1 - x, 0)

class MscaleDNN1(nn.Module):
    hidden: int
    depth: int
    out_dim: int
    scales: tuple                                              # e.g. (1,2,4,8,16,32)

    @nn.compact
    def __call__(self, x):
        N = len(self.scales)
        assert self.hidden % N == 0
        group = self.hidden // N
        # K broadcasts each scale across its slice of hidden units; treated as a constant.
        K = jnp.repeat(jnp.asarray(self.scales, dtype=x.dtype), group)   # (hidden,)
        h = nn.Dense(self.hidden, name="lin0")(x) * K[None, :]
        h = sReLU(h)
        for i in range(self.depth - 1):
            h = sReLU(nn.Dense(self.hidden, name=f"lin{i+1}")(h))
        return nn.Dense(self.out_dim, name="lin_out")(h)

net = MscaleDNN1(hidden=120, depth=4, out_dim=1, scales=(1, 2, 4, 8, 16, 32))
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

def ritz_loss(params, xq, wq, x_bc, g_bc, eps, kappa, rho_fn, beta=500.0):
    def u_single(p, xi): return net.apply(p, xi[None])[0, 0]
    u  = jax.vmap(lambda xi: u_single(params, xi))(xq)
    gu = jax.vmap(lambda xi: jax.grad(u_single, argnums=1)(params, xi))(xq)
    bulk = (0.5 * eps * jnp.sum(gu ** 2, axis=1)
            + kappa ** 2 * (jnp.cosh(u) - 1)
            - rho_fn(xq) * u)
    L_int = jnp.sum(bulk * wq)
    L_bc  = beta * jnp.mean((net.apply(params, x_bc)[:, 0] - g_bc) ** 2)
    return L_int + L_bc

@jax.jit
def train_step(params, opt_state, batch):
    grads = jax.grad(ritz_loss)(params, *batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

Recommended: scales `a_i = 2^(i-1)` with `N = 5-8`; hidden width `120-240` (must be divisible by `N`); depth 4-6; quadratic B-spline activation for smooth derivatives; Adam `lr=1e-3` then L-BFGS; `beta = 500`. Use MscaleDNN-1 for parameter efficiency, MscaleDNN-2 for cleanest scale separation.

## Results
On 2-D and 3-D Poisson-Boltzmann over molecular vdW surfaces (singular cusps / self-intersections), ring-shaped and multi-hole domains: MscaleDNN reduces relative L2 error by 1-2 orders of magnitude versus a same-size plain MLP, with similar wall time. Resolves both low- and high-frequency content uniformly, unlike the plain MLP which stalls on high-frequency modes.
