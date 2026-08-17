---
slot: 98
title: "A Wachspress-based transfinite formulation for exactly enforcing Dirichlet boundary conditions on convex polygonal domains in physics-informed neural networks"
authors: [N. Sukumar, Ritwick Roy]
year: 2026
venue: Computational Mechanics (arXiv:2601.01756)
gitrepo: ""
---

## TL;DR
For PINNs / deep Ritz on convex polygons, build a trial function `u(x) = g(x) + (N(x) - L[N](x))` where `g` is Randrianarivony's Wachspress-based transfinite interpolant lifting the prescribed boundary function B to the interior. Wachspress barycentric coordinates `lambda_i(x)` are also used as a geometric feature map to the network, eliminating the unbounded-Laplacian pathology of approximate-distance-function (ADF) ansatzes at polygon vertices.

## Problem
Hard-imposed Dirichlet BCs improve PINN accuracy vs soft penalties. The Sukumar-Srivastava ADF ansatz `u = g + phi N` requires `phi = 0, dphi/dn = 1` on every edge; at a polygon vertex these two requirements are inconsistent and `|grad^2 phi|` diverges, forcing one to shrink the collocation domain by `delta = 1e-2`. Also, the multiplicative `phi N` couples boundary stiffness and interior, hurting gradient flow.

## Method
Wachspress coordinates on a convex `n`-gon (closed form via Meyer et al.):
$$
\lambda_i(x) = \frac{w_i(x)}{\sum_j w_j(x)},\quad
w_i(x) = \frac{\det(n_{i-1}, n_i)}{h_{i-1}(x)\,h_i(x)}
$$
They satisfy `lambda_i >= 0`, `sum lambda_i = 1`, `sum lambda_i x_i = x`, Kronecker delta at vertices, and on edge `e_i` only `lambda_i + lambda_{i+1} = 1`. They are C-infinity smooth, so derivatives are bounded everywhere (including vertices).

Boundary function `B(lambda)` parametrized by Wachspress coords. The Randrianarivony lifting operator on a quadrilateral is
$$
g(\lambda) = \lambda_1[\alpha_1(\lambda_2)+\alpha_4(1-\lambda_4)-\alpha_1(0)]
+ \lambda_2[\alpha_2(\lambda_3)+\alpha_1(1-\lambda_1)-\alpha_2(0)]
$$
$$
+\,\lambda_3[\alpha_3(\lambda_4)+\alpha_2(1-\lambda_2)-\alpha_3(0)]
+ \lambda_4[\alpha_4(\lambda_1)+\alpha_3(1-\lambda_3)-\alpha_4(0)]
$$
which restricts to `alpha_i` on edge `e_i`. For general `n`-gon use the same pattern (eq. 17 in the paper).

Trial function (TFI) for PINN:
$$
u_\theta^{TFI}(x;\theta) = g(\lambda(x)) + \big[N_\theta(\lambda(x);\theta) - L[N_\theta]\big]
$$
where `L[N_theta]` substitutes the same g-formula with `B` replaced by `N_theta`'s boundary restriction. By construction `u_theta = g` on partial-Omega.

```python
import jax, jax.numpy as jnp
import flax.linen as nn

def wachspress(x, vertices):                       # x:[B,2], vertices:[n,2] CCW convex
    n = vertices.shape[0]
    edges   = jnp.roll(vertices, -1, axis=0) - vertices                 # [n,2]
    normals = jnp.stack([edges[:, 1], -edges[:, 0]], axis=-1)
    normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
    # h_i(x) = (x - v_i) . n_i
    h = jnp.einsum("bd,nd->bn", x[:, None, :].squeeze(1) - vertices[None, :, :].squeeze(0),
                   normals) if False else (
        jnp.stack([(x - vertices[i]) @ normals[i] for i in range(n)], axis=-1))
    w = jnp.stack([jnp.linalg.det(jnp.stack([normals[(i-1) % n], normals[i]]))
                   / (h[:, (i-1) % n] * h[:, i]) for i in range(n)], axis=-1)
    return w / w.sum(-1, keepdims=True)

def alpha_param(i, t, B_fn, vertices):
    n = vertices.shape[0]
    return B_fn((1.0 - t)[:, None] * vertices[i] + t[:, None] * vertices[(i + 1) % n])

def lift_g(lam, B_fn, vertices):
    n = vertices.shape[0]; g = 0.0
    for i in range(n):
        ip1, im1 = (i + 1) % n, (i - 1) % n
        a_here = alpha_param(i,   lam[:, ip1], B_fn, vertices)
        a_prev = alpha_param(im1, 1.0 - lam[:, im1], B_fn, vertices)
        v_val  = B_fn(vertices[i:i+1])
        g = g + lam[:, i] * (a_here + a_prev - v_val.squeeze(0))
    return g

class MLP(nn.Module):
    hidden: int = 64; depth: int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x).squeeze(-1)

net = MLP()                                    # eats Wachspress coords directly
n_v = vertices.shape[0]
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, n_v)))

def trial(params, x, vertices, B_fn):
    lam = wachspress(x, vertices)
    def N_on_boundary(p):
        return net.apply(params, wachspress(p, vertices))
    g   = lift_g(lam, B_fn, vertices)
    L_N = lift_g(lam, N_on_boundary, vertices)
    return g + (net.apply(params, lam) - L_N)

def pde_loss(params, x, vertices, B_fn, f_fn):     # Poisson: -Δu = f
    def u_scalar(xv): return trial(params, xv[None], vertices, B_fn).squeeze()
    lap = jax.vmap(lambda xv: jnp.trace(jax.hessian(u_scalar)(xv)))(x)
    return jnp.mean((-lap - f_fn(x))**2)
```

Hyperparameters: tanh MLPs of depth 4 x 64-128, Adam then L-BFGS, collocation points sampled uniformly across the full polygon (no delta-shrink needed because Wachspress Laplacian is bounded), Wachspress input layer replaces raw `(x,y)`.

## Results
Demonstrated on Poisson (linear and one nonlinear, including oscillatory BCs), an inverse heat-source problem, a parameterized-geometry Poisson, and the Eikonal equation on triangle, square, quadrilateral, pentagon, octagon. The trial function exactly matches Dirichlet data; loss is the single PDE-residual (or deep-Ritz energy) term. Collocation can be sampled up to and including the boundary because Laplacian of the ansatz is bounded; the previously needed `delta = 1e-2` exclusion is removed. Wachspress coordinates as input features enable a single trained network to generalize over parametrized convex geometries.
