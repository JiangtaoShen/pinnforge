---
slot: 70
title: "Separable Physics-Informed Neural Networks (SPINN)"
authors: [Junwoo Cho, Seungtae Nam, Hyunmo Yang, Youngjoon Hong, Eunbyung Park]
year: 2023
venue: NeurIPS 2023 (arXiv:2306.15969)
gitrepo: ""
---

## TL;DR
Replace the monolithic d-input MLP with **d per-axis MLPs**, each `f^{(i)} : R -> R^r`, combined by an outer product / element-wise multiply and rank-`r` sum. A factorizable tensor-grid of collocation points reduces evaluations from `O(N^d)` to `O(Nd)`, and **forward-mode AD** computes residual derivatives in `O(Nd)` JVPs. Result: solve (3+1)-D Navier-Stokes on a single GPU using `>1e7` collocation points; 62x wall-clock speedup vs causal PINN on (2+1)-D NS.

## Problem
Conventional MLP PINNs evaluate the network at every `(x_1, ..., x_d)` point individually; for high-dimensional PDEs `N^d` collocation points are needed for accuracy, but compute and memory grow as `O(N^d * f)`. Even with reverse-mode AD, Hessians/Laplacians on a single GPU cap out around 24-64 points per axis in 3-D, far from the resolution needed for turbulent NS.

## Method
The solution ansatz is a rank-`r` CP-decomposition:
$$
\hat u(x_1, \ldots, x_d) = \sum_{j=1}^{r} \prod_{i=1}^{d} f_j^{(\theta_i)}(x_i)
$$
Each `f^{(\theta_i)} : R -> R^r` is a small MLP (tanh) over scalar `x_i`. With `N` samples per axis, this defines a value at every point of the lattice `N^d` while only requiring `N d` network evaluations and `N r d` features. Universal approximation holds (Thm 1).

**Forward-mode AD** for `partial u / partial x_i`: only `f^{(i)}` depends on `x_i`, so
$$
\frac{\partial \hat u}{\partial x_i} = \sum_{j=1}^{r} \frac{\partial f_j^{(\theta_i)}(x_i)}{\partial x_i} \prod_{k \ne i} f_j^{(\theta_k)}(x_k)
$$
One JVP per axis gives all `r` partials at all `N` samples — total `N d` JVPs for the full gradient; same trick for higher-order derivatives (just chain JVPs). The `N^d` solution tensor is materialized by outer products in memory-efficient form (only `N r d` floats).

Architectures: identical small MLPs per axis (e.g. 5 hidden layers x 64 tanh), `r ∈ [64, 256]`. Compatible with the improved-MLP (Wang 2021) backbone -> "SPINN-mod" for stiff problems. PINN loss is unchanged; collocation lattice can grow to `N=128..256` per axis on 24GB GPU.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
from string import ascii_lowercase

class AxisBody(nn.Module):
    hidden: int = 64
    depth:  int = 5
    rank:   int = 128

    @nn.compact
    def __call__(self, x):              # x: [N, 1]
        h = x
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.hidden)(h))
        return nn.Dense(self.rank)(h)   # [N, r]

class SPINN(nn.Module):
    d:      int = 3
    hidden: int = 64
    depth:  int = 5
    rank:   int = 128

    def setup(self):
        self.bodies = [AxisBody(self.hidden, self.depth, self.rank,
                                name=f"axis_{i}") for i in range(self.d)]

    def __call__(self, axes):                       # axes: list of d arrays [N, 1]
        feats = [self.bodies[i](axes[i]) for i in range(self.d)]   # each [N, r]
        idx   = ascii_lowercase[: self.d]
        expr  = ",".join(f"{c}r" for c in idx) + "->" + "".join(idx)
        return jnp.einsum(expr, *feats)             # [N]*d

# Forward-mode AD of u w.r.t. x_i: differentiate body_i, keep others fixed
def du_dxi(params, apply_fn, axes, i, order=1):
    """Return d^order u / d x_i^order as a tensor of shape [N]*d."""
    # Pre-compute features for k != i; differentiate body i via repeated JVP
    bodies = apply_fn.__self__.bodies if False else None  # placeholder
    def body_i(z): return apply_fn({"params": params["params"]}, [z]*0 + [z])  # see note
    # Idiomatic: write a helper that takes (params, axes) and returns features.
    def feats_with_xi(xi):
        new_axes = [axes[k] if k != i else xi for k in range(len(axes))]
        return apply_fn(params, new_axes, method=lambda m, a: [m.bodies[k](a[k])
                                                                for k in range(m.d)])
    # Iterated JVP (forward-mode)
    f = feats_with_xi
    tangent = jnp.ones_like(axes[i])
    for _ in range(order):
        f, _ = (lambda fn: (lambda z: jax.jvp(fn, (z,), (tangent,))[1], None))(f)
    feat_list = f(axes[i])
    idx  = ascii_lowercase[: len(axes)]
    expr = ",".join(f"{c}r" for c in idx) + "->" + "".join(idx)
    return jnp.einsum(expr, *feat_list)

def pinn_loss(params, apply_fn, axes, f_fn):
    U   = apply_fn(params, axes)
    Uxx = sum(du_dxi(params, apply_fn, axes, i, order=2) for i in range(len(axes)))
    return jnp.mean((Uxx - f_fn(axes))**2)
```

For time-dependent PDEs, treat `t` as one of the axes. For non-rectangular domains use a level-set mask or weight the residual by an indicator (still factorizable for cubes).

## Results
On 3-D diffusion, Helmholtz, (2+1)-D and (3+1)-D Klein-Gordon: SPINN (often with modified-MLP body) achieves 1-2 orders of magnitude lower relative L2 than vanilla PINN at 52-62x faster wall-clock and 29x lower memory; FLOPs for residual derivatives drop 1394x. On (2+1)-D chaotic Navier-Stokes, SPINN matches causal-PINN accuracy in 9 min vs 10 h on the same single GPU, and solves a (3+1)-D Navier-Stokes baseline that prior PINNs could not.
