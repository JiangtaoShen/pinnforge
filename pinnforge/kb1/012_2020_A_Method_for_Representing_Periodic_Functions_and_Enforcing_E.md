---
slot: 12
title: "A Method for Representing Periodic Functions and Enforcing Exactly Periodic Boundary Conditions with Deep Neural Networks"
authors: [Suchuan Dong, Naxian Ni]
year: 2020
venue: "Journal of Computational Physics (arXiv:2007.07442)"
gitrepo: ""
---

## TL;DR
Insert a single specially-designed *periodic layer* as the second layer of an MLP. Any subsequent network on top automatically inherits *exact* periodicity (to machine precision) of the output and *all* its derivatives (C-infinity case), or up to derivative order k (C^k case). Eliminates the BC penalty for periodic problems entirely.

## Problem
Penalty-based enforcement of periodic BCs only matches the function (and maybe first derivative) approximately and tuning the penalty weight is brittle; higher-derivative periodicity is essentially unattainable by penalties; existing "sinusoidal first layer" tricks miss higher-frequency content.

## Method

### A. C-infinity periodic layer
Use a 1-D smooth periodic function `v(x) = sigma(A cos(omega x + phi) + c)` where `omega = 2 pi / L` is fixed (the desired period), `A, phi, c` are trainable, and `sigma` is a nonlinear activation (tanh/sigmoid) which generates harmonics so the layer can represent any frequency content with period `L`.

Build a layer `L_p(m, n)` with `m` such "sine-bank" units feeding `n` output neurons:
$$
v_i(x) = \sigma(A_i \cos(\omega x + \phi_i) + c_i),\quad i=1..m
$$
$$
q_j(x) = \sigma\!\Big(\sum_{i=1}^{m} W_{ij}\,v_i(x) + B_j\Big),\quad j=1..n
$$
Insert `L_p` immediately after the input. Stack ordinary `Dense(tanh)` layers afterwards. The full network output `u(x) = f_DNN(q(x))` then satisfies `u(a) = u(b)` and `u^{(l)}(a) = u^{(l)}(b)` for all `l`, by Lemma 2.1: any smooth composition `f(v(x))` of a periodic `v` is periodic in all orders.

2-D extension on rectangular cell `[a_1, b_1] x [a_2, b_2]` with periods `L_1, L_2`: build separate periodic units per axis and combine:
$$
q_j(x_1, x_2) = \sigma\!\Bigl(\sum_{i} W^{(1)}_{ij} v_{1i}(x_1) + \sum_{i} W^{(2)}_{ij} v_{2i}(x_2) + B_j\Bigr)
$$
Generalises to any dimension.

### B. C^k periodic layer (finite-order periodicity)
Replace `cos(omega x + phi)` by a generalised Hermite interpolation polynomial of degree `2k+1` that satisfies `H^{(l)}(a) = H^{(l)}(b)` for `l = 0..k`. Same outer wiring `sigma(H_i(x)) -> Dense`. Gives exactly C^k periodicity (useful when the solution has only finite smoothness across the cell boundary).

JAX (flax.linen):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import math

class PeriodicLayerCInf(nn.Module):
    """1-D C-infinity periodic layer: m sine-units per input dim, n outputs."""
    in_dim: int
    m: int
    n: int
    L: float
    act: callable = jnp.tanh

    @nn.compact
    def __call__(self, x):                                  # x: (N, in_dim)
        omega = 2 * math.pi / self.L
        A   = self.param("A",   nn.initializers.normal(1.0), (self.in_dim, self.m))
        phi = self.param("phi", nn.initializers.normal(1.0), (self.in_dim, self.m))
        c   = self.param("c",   nn.initializers.zeros,       (self.in_dim, self.m))
        ang = omega * x[:, :, None] + phi                   # (N, in_dim, m)
        v = self.act(A * jnp.cos(ang) + c)                  # (N, in_dim, m)
        v = v.reshape(x.shape[0], -1)                       # (N, in_dim*m)
        return self.act(nn.Dense(self.n)(v))

class PeriodicMLP(nn.Module):
    in_dim: int
    hidden: int
    depth: int
    out_dim: int
    L: float
    m: int = 10
    @nn.compact
    def __call__(self, x):
        x = PeriodicLayerCInf(self.in_dim, self.m, self.hidden, self.L)(x)
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(self.out_dim)(x)

net = PeriodicMLP(in_dim=1, hidden=40, depth=4, out_dim=1, L=2 * math.pi, m=10)
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 1)))
# Loss is just MSE_residual + MSE_IC (no MSE_periodic_BC).
```

Recommended: `m = 5-15` periodic units, hidden width 30-50, depth 4-6, tanh; for 2-D PDE on a periodic cell use 2-D variant with separate `m` units per axis. For Helmholtz, diffusion, wave equations with periodic BCs.

## Results
On 1-D Helmholtz, 2-D Helmholtz, diffusion, and wave equations with C-infinity periodicity, the resulting network satisfies periodicity to machine precision (`u(b) - u(a)` and all measured derivatives match to ~14 significant digits); solution accuracy is comparable to a penalty-PINN (1-D Helmholtz, fixed penalty `theta_bc = 10`) while enforcing periodicity exactly rather than approximately. C^k variant verified up to k = 2 (wave equation) with machine-precision matching of the first k derivatives only.
