---
slot: 036
title: "Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks"
authors: [N. Sukumar, Ankit Srivastava]
year: 2021
venue: "CMAME (arXiv:2104.08426)"
gitrepo: ""
---

## TL;DR
Build an analytic Approximate Distance Function (ADF) phi(x) to the boundary using R-functions (constructive solid geometry) or mean value coordinates, then write the trial as u_theta(x) = g(x) + phi(x) u_hat_theta(x). Dirichlet BC exact a priori, loss reduces to PDE residual only. Generalizes to inhomogeneous Dirichlet, Neumann, Robin and mixed BCs over complex 2D/3D/4D domains.

## Problem
Soft-BC PINNs trade off interior residual vs boundary residual via lambda — a multi-objective optimization whose Pareto solution depends on weights, and accuracy degrades on complex geometries. Earlier hard-BC methods either need a separate boundary network or are limited to simple shapes.

## Method
**ADF construction.** For boundary partition into line segments / curves {Gamma_i}, each has an analytic phi_i (e.g. line: f_i = signed distance to infinite line, trimmed by disk t_i, then phi_i = sqrt(f_i^2 + ((t_i^2+f_i^4)^{1/2} - t_i)^2 / 4)). Combine via R-conjunction (intersect) and R-disjunction (union):
$$
\omega_1\vee\omega_2 = \omega_1 + \omega_2 + \sqrt{\omega_1^2 + \omega_2^2},\quad
\omega_1\wedge\omega_2 = \omega_1 + \omega_2 - \sqrt{\omega_1^2 + \omega_2^2}
$$
Result: phi(x) > 0 in Omega, phi = 0 on dOmega, normalized so partial phi/partial nu = 1.

**Dirichlet trial (homogeneous u=0 on dOmega):**
$$
u_\theta(x) = \phi(x)\,\hat u_\theta(x)
$$

**Inhomogeneous Dirichlet u = g on dOmega** via Kantorovich/transfinite-interpolation:
$$
u_\theta(x) = g(x - \phi(x)\nabla\phi(x)) + \phi(x)\,\hat u_\theta(x)
$$
The shift x - phi grad phi maps interior points to their boundary projection.

**Mixed BC** (Dirichlet g_1 on Gamma_1, Robin du/dn + c u = h on Gamma_2): compose distance functions phi_1 (Gamma_1) and phi_2 (Gamma_2 only), and use partition-of-unity transfinite interpolation (Rvachev-Sheiko).

Loss is now purely interior residual: L = (1/N_r) sum |N[u_theta](x_r) - f(x_r)|^2.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

def adf_line_segment(x, x1, x2):
    L  = jnp.linalg.norm(x2 - x1)
    xc = 0.5*(x1 + x2)
    f  = ((x[0]-x1[0])*(x2[1]-x1[1]) - (x[1]-x1[1])*(x2[0]-x1[0])) / L
    t  = ((L/2)**2 - jnp.sum((x - xc)**2)) / L
    varphi = jnp.sqrt(t**2 + f**4)
    return jnp.sqrt(f**2 + ((varphi - t)/2)**2 + 1e-12)

def r_conjunction(phis):                    # intersection (all sides)
    s  = jnp.sum(phis)
    sq = jnp.sum(phis**2)
    return s - jnp.sqrt(sq + 1e-12)

def phi_polygon(x, vertices):
    phis = jnp.stack([adf_line_segment(x, vertices[i], vertices[(i+1)%len(vertices)])
                      for i in range(len(vertices))])
    return r_conjunction(phis)

class UHat(nn.Module):
    width: int = 40; depth: int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.width)(x))
        return nn.Dense(1)(x)

def trial_u(params, apply_fn, phi_fn, g_fn, x):
    phi = phi_fn(x)
    uhat = apply_fn(params, x)[0]
    if g_fn is None:
        return phi * uhat                   # homogeneous Dirichlet
    # transfinite: u = g(x - phi*grad_phi) + phi*uhat
    grad_phi = jax.grad(phi_fn)(x)
    x_proj   = x - phi * grad_phi
    return g_fn(x_proj) + phi * uhat

def loss_interior(params, apply_fn, phi_fn, g_fn, pde_op, x_int, f_src):
    def u_at(x): return trial_u(params, apply_fn, phi_fn, g_fn, x)
    r = jax.vmap(lambda x: pde_op(u_at, x) - f_src(x))(x_int)
    return jnp.mean(r**2)

net = UHat()
params = net.init(jax.random.PRNGKey(0), jnp.zeros(2))
opt = optax.adam(1e-3); state = opt.init(params)

@jax.jit
def step(params, state, x_int):
    g = jax.grad(loss_interior)(params, net.apply, phi_fn, g_fn, pde_op, x_int, f_src)
    upd, state = opt.update(g, state, params)
    return optax.apply_updates(params, upd), state
```

Recommended: tanh MLP 4x40, Adam then L-BFGS, mean-value coordinates ADF for curved boundaries, R-equivalence composition for multiply-connected domains.

## Results
On Poisson over L-shape, plate-with-hole, biharmonic (Kirchhoff plate), Eikonal on curved domains, and 4-D Poisson on the hypercube, exact-BC PINN consistently matches or beats soft-BC PINN by 1-3 orders of magnitude in L2 error with simpler optimization (single-objective).
