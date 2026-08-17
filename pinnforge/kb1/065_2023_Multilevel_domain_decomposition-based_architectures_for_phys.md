---
slot: 65
title: "Multilevel domain decomposition-based architectures for physics-informed neural networks"
authors: [Victorita Dolean, Alexander Heinlein, Siddhartha Mishra, Ben Moseley]
year: 2023
venue: Computer Methods in Applied Mechanics and Engineering (CMAME)
gitrepo: "https://github.com/benmoseley/FBPINNs/tree/multilevel-paper"
doi: 10.1016/j.cma.2024.117116
---

## TL;DR
**Multilevel FBPINNs** stack several overlapping domain decompositions at increasing resolutions; each subdomain hosts a small MLP that is restricted by a smooth window function (partition of unity), and the global solution is the sum across all subdomains across all levels. Inspired by multilevel Schwarz methods, this enables global communication between fine subdomains via the coarse-level network, dramatically improving accuracy on high-frequency and multi-scale PDEs.

## Problem
For high-frequency or multi-scale solutions a single PINN's spectral bias forces ever-larger networks and collocation sets; cost grows super-linearly. Single-level FBPINN (Moseley et al. 2022) helps by placing small MLPs in overlapping subdomains, but with many subdomains information cannot propagate globally across the whole domain — a known issue for classical one-level Schwarz methods, fixed in numerical analysis by coarse levels.

## Method
Decompose `Omega` into `L` levels of overlapping subdomains. Without loss of generality `J^{(1)} = 1` (a single coarse subdomain covering `Omega`) and `J^{(1)} < J^{(2)} < ... < J^{(L)}`. At every level `l` and subdomain `j` place a small MLP `v_j^{(l)}(x; theta_j^{(l)})` with **per-subdomain coordinate normalization** to `[-1, 1]` — this rescales each local problem to lower effective frequency, easing spectral bias. Each `v_j^{(l)}` is multiplied by a smooth window `omega_j^{(l)}(x)` supported on `Omega_j^{(l)}`; the windows form a partition of unity per level.

The multilevel FBPINN solution is
$$
u(x;\theta) = \sum_{l=1}^{L}\sum_{j=1}^{J^{(l)}} \omega_j^{(l)}(x)\,v_j^{(l)}(x;\theta_j^{(l)})
$$
which is plugged into the standard PINN loss (soft BC) or, preferably, the hard-BC form `Cu` with `C(x)` a known function that zeros at the boundary:
$$
\mathcal L(\theta) = \frac{1}{N}\sum_{i=1}^{N}\!\Big(\mathcal N\!\big[C\cdot u(\cdot;\theta)\big](x_i) - f(x_i)\Big)^2
$$

A common window choice on a rectangular subdomain is the product of 1-D `cos^2` ramps. Because only subdomains containing `x_i` contribute, evaluating the sum at one collocation point is `O(C)` not `O(J)` (C = avg overlap multiplicity), so the cost stays linear in `J`.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import math

class SubdomainMLP(nn.Module):
    hidden: int = 32
    depth:  int = 2
    x0:     tuple = (0.0,)
    half:   tuple = (1.0,)

    @nn.compact
    def __call__(self, x):
        x0   = jnp.asarray(self.x0)
        half = jnp.asarray(self.half)
        z    = (x - x0) / half                  # normalize to [-1, 1]
        for _ in range(self.depth):
            z = nn.tanh(nn.Dense(self.hidden)(z))
        return nn.Dense(1)(z)

def cos2_window(x, x0, half):
    x0   = jnp.asarray(x0); half = jnp.asarray(half)
    z = (x - x0) / half
    inside = jnp.all(jnp.abs(z) < 1.0, keepdims=True).astype(x.dtype)
    w = jnp.prod(jnp.cos(0.5 * math.pi * jnp.clip(z, -1.0, 1.0))**2,
                 keepdims=True)
    return w * inside

class MultilevelFBPINN(nn.Module):
    geom: tuple                                  # tuple of tuples of (x0, half)
    hidden: int = 32
    depth:  int = 2

    @nn.compact
    def __call__(self, x):
        u = jnp.zeros((1,))
        for l, lvl in enumerate(self.geom):
            for j, (x0, half) in enumerate(lvl):
                mlp = SubdomainMLP(self.hidden, self.depth, tuple(x0), tuple(half),
                                   name=f"L{l}_S{j}")
                w   = cos2_window(x, x0, half)
                u   = u + w * mlp(x)
        return u

# Hard-BC ansatz for u(0)=u(1)=0 on [0,1]:
def Cu(params, apply_fn, x):
    return x[..., 0:1] * (1 - x[..., 0:1]) * apply_fn(params, x)

import optax
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, X_coll):
    def loss(p):
        r = jax.vmap(lambda x: pde_residual_hardBC(p, apply_fn, x) - f(x))(X_coll)
        return jnp.mean(r**2)
    grads = jax.grad(loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

Recommended sizes: small per-subdomain MLPs (2 layers x 16-32 tanh); `L = 2..3` levels with `J^{(l)}` doubling (e.g. 1 -> 4 -> 16 in 1-D). Use ~10-20 collocation points per fine subdomain.

## Results
Across high-frequency 1-D/2-D Poisson, multi-scale Helmholtz, and a 2-D wave problem with strong/weak scaling tests, multilevel FBPINN matches reference solutions with relative L2 ~ `1e-3..1e-5` while single PINN, Fourier-feature PINN, and SA-PINN saturate around `1e-1..1e-2`. As `J^{(L)}` grows, vanilla FBPINN degrades (lack of global coupling) but multilevel FBPINN keeps scaling — the coarse network supplies global communication exactly as in classical multilevel Schwarz.
