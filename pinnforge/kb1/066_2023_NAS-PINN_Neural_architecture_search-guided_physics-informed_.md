---
slot: 66
title: "NAS-PINN: Neural architecture search-guided physics-informed neural network for solving PDEs"
authors: [Yifan Wang, Linlin Zhong]
year: 2023
venue: Journal of Computational Physics (arXiv:2305.10127)
gitrepo: ""
---

## TL;DR
Use DARTS-style differentiable NAS to **automatically choose the depth and per-layer width** of an MLP for a given PDE. A super-net relaxes the discrete architecture choice with continuous gates `(a_1, a_2)` (skip/keep) and width masks `g_k` (one-hot relaxed); bi-level optimization alternates standard PINN loss to update `theta`, and a small supervised MSE on a sparse reference to update `alpha`. Findings: shallow + wide beats deep + narrow on Poisson/Advection; residual connections help on complex domains.

## Problem
PINN architectures are hand-picked (typically 4-6 hidden layers, equal widths) — no theory tells you whether to go deeper, wider, or add skip-connections. Grid search over `(depth, width)` is slow and limited; standard NAS uses discrete RL and ignores PDE-specific loss.

## Method

### A. Super-net with masks
Each candidate hidden layer is parameterized to handle (a) skip vs keep, (b) variable number of neurons via fixed-shape padding plus binary masks. With max width `k` and width options `n_1 < n_2 < ... < n_R = k`, define mask `m_r in {0,1}^k` with first `n_r` ones. One layer output:
$$
y = a_1 \cdot x + a_2 \cdot \sigma(Wx + b)\cdot\sum_{r=1}^{R} g_r\, m_r
$$
with softmax-relaxed gates `a = softmax(a_raw)`, `g = softmax(g_raw)`. Stack `L_max` such layers; learnable architecture vector `alpha = {a_l, g_l}_{l=1..L_max}`. After training, discretize: keep layer `l` if `a_2 > a_1` (else skip); pick width `argmax g_l`. If `|a_1 - a_2|` is small, retain a "mixed layer" `y = a_1 x + a_2 sigma(Wx + b)*m_{r*}` — emerges as a learned residual connection.

### B. Bi-level optimization
- **Inner loop (n_inner steps)**: hold `alpha` fixed, update PINN weights `theta = (W,b)` by Adam on the standard PDE+BC+IC loss.
- **Outer step (every n_inner)**: hold `theta` fixed, update `alpha` by Adam on a small **supervised MSE** against a tiny reference set (a few hundred points from analytic/numerical solution).
$$
\min_\alpha \frac{1}{n}\sum_i (\hat u(x_i;\theta^*(\alpha), \alpha) - u(x_i))^2 \quad\text{s.t.}\quad \theta^*(\alpha) = \arg\min_\theta L_{PINN}(\theta, \alpha)
$$

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class SuperLayer(nn.Module):
    in_d:   int
    widths: tuple = (30, 50, 70)

    @nn.compact
    def __call__(self, x):
        k = max(self.widths)
        lin   = nn.Dense(k)
        proj  = nn.Dense(self.in_d)
        a_raw = self.param("a_raw", nn.initializers.zeros, (2,))
        g_raw = self.param("g_raw", nn.initializers.zeros, (len(self.widths),))
        masks = jnp.stack([
            jnp.concatenate([jnp.ones(w), jnp.zeros(k - w)])
            for w in self.widths
        ], axis=0)                                                # [R, k]
        h = nn.tanh(lin(x))
        g = jax.nn.softmax(g_raw)
        m = jnp.sum(g[:, None] * masks, axis=0)                   # [k]
        h = proj(h * m)
        a = jax.nn.softmax(a_raw)
        return a[0] * x + a[1] * h

class NASPINN(nn.Module):
    in_d:   int
    out_d:  int
    L_max:  int = 5
    widths: tuple = (30, 50, 70)

    @nn.compact
    def __call__(self, x):
        k = max(self.widths)
        h = nn.tanh(nn.Dense(k)(x))
        for _ in range(self.L_max):
            h = SuperLayer(in_d=k, widths=self.widths)(h)
        return nn.Dense(self.out_d)(h)

def split_arch(params):
    """Return (arch_params, net_params) by name."""
    flat = jax.tree_util.tree_leaves_with_path(params)
    arch, net = {}, {}
    for path, v in flat:
        key = "/".join(str(p) for p in path)
        (arch if "raw" in key else net)[key] = v
    return arch, net   # use optax.masked for selective updates in practice

opt_w = optax.adam(1e-3)                                # weights
opt_a = optax.adam(3e-4)                                # architecture
state_w = opt_w.init(params); state_a = opt_a.init(params)

@jax.jit
def inner_step(params, state_w, X_r, X_b, U_b):
    grads = jax.grad(lambda p: pinn_loss(p, X_r, X_b, U_b))(params)
    # In practice mask out arch params via optax.masked
    updates, state_w = opt_w.update(grads, state_w, params)
    return optax.apply_updates(params, updates), state_w

@jax.jit
def outer_step(params, state_a, X_ref, U_ref):
    def mse(p): return jnp.mean((jax.vmap(apply_fn, in_axes=(None, 0))(p, X_ref) - U_ref)**2)
    grads = jax.grad(mse)(params)
    updates, state_a = opt_a.update(grads, state_a, params)
    return optax.apply_updates(params, updates), state_a

for outer in range(n_outer):
    for _ in range(n_inner):
        params, state_w = inner_step(params, state_w, X_r, X_b, U_b)
    params, state_a = outer_step(params, state_a, X_ref, U_ref)
# Discretize after training:
#   keep layer l if a_l[1] > a_l[0]; width = widths[argmax g_l]
```

Hyper-params: `L_max = 5-7`, `widths = (30, 50, 70)` or `(10, 30, ..., 110)`, `n_inner = 50..200`, `n_outer = 50..200`, Adam(1e-3 for weights, 3e-4 for arch). Architecture-stage uses a slightly denser collocation/reference set (e.g. 1000 / 200).

## Results
Across 2-D Poisson (square, circle, L-shape, flower), 1-D Burgers, and 1-D Advection, NAS-PINN finds architectures that outperform every architecture in the discrete search grid and the Bayesian-optimization baseline SMAC, with 25% less search time. Headline finding: a 2-layer (50, 70) network beats deep 5-layer architectures on Poisson; on irregular domains, residual mixed layers emerge automatically and reduce error 5-10x. Lesson: shallow + wide + residual is often optimal for PINN; depth alone hurts.
