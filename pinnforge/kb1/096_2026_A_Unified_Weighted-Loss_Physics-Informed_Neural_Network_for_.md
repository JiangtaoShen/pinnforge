---
slot: 96
title: "A Unified Weighted-Loss Physics-Informed Neural Network for Boundary Layer Problems in Singularly Perturbed PDEs"
authors: [Wei-Fan Hu, Shi-Xiang Zhong, Po-Wen Hsieh, Chung-Kai Chen, Te-Sheng Lin]
year: 2026
venue: arXiv:2603.29249
gitrepo: ""
---

## TL;DR
Build a PINN whose output is the sum of a regular MLP `u_r(x)` plus singular MLPs `u_{s,L}(phi_L(x)/eps)` and `u_{s,R}(phi_R(x)/eps)` driven by epsilon-scaled signed-distance level sets. Weight the PDE residual by the squared distance to the boundary to balance the `O(1/eps^2)` interior loss against `O(1)` boundary loss. Achieves 1e-7 to 1e-8 errors for `eps = 1e-10`.

## Problem
Singularly perturbed PDEs (reaction-diffusion `-eps^2 u'' + u = f`, convection-diffusion-reaction `-eps u'' + a u' + gamma u = f`, Poisson-Boltzmann, coupled systems) have boundary layers of width `O(eps)`. Vanilla PINNs cannot represent the `O(1/eps)` first / `O(1/eps^2)` second derivatives, and BL-PINN/SD-PINN require explicit asymptotic matching of multiple sub-networks. The authors want one architecture, one loss, for every variant.

## Method
Solution ansatz (1-D, `Omega = [a,b]`):
$$
u(x) = u_r(x) + u_{s,L}\!\Big(\frac{x-a}{\varepsilon}\Big) + u_{s,R}\!\Big(\frac{x-b}{\varepsilon}\Big)
$$
because `sigma((x-x0)/eps)` has derivative `(1/eps) sigma(1-sigma)` localized in a band of width `O(eps)`. In 2-D regular domains, use a multiplicative dimension-by-dimension split with four edge networks; in irregular domains a single `u_s(x, y, phi(x,y)/eps)` driven by a level-set function with `|grad phi| = O(1)`.

Weighted loss with boundary-distance weighting (only needed for convection-dominated cases; `w == 1` works for reaction-diffusion / Poisson-Boltzmann):
$$
w(x) = \big(\min(x-a, b-x)\big)^2 \;\; (1\text{D}),\quad
w(x,y) = \big(\min(x-a, b-x, y-c, d-y)\big)^2
$$
$$
\mathcal J(\theta) = \frac1m\sum_{i=1}^{m} w(x_i)\,|L_\varepsilon u(x_i) - f(x_i)|^2 + \frac1{m_b}\sum_{j=1}^{m_b}|u(x_j^b)-g(x_j^b)|^2
$$
Inside boundary layers `L_eps u` is `O(1/eps)` so the residual squared is `O(1/eps^2)`; multiplying by `w = O(eps^2)` rebalances both regions to `O(1)`.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import jaxopt

class Block(nn.Module):
    hidden: int = 50; depth: int = 1
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jax.nn.sigmoid(nn.Dense(self.hidden)(x))
        return x

class BLPINN1D(nn.Module):
    a: float = 0.0; b: float = 1.0; eps: float = 1e-10; hidden: int = 50
    @nn.compact
    def __call__(self, x):
        phiL = (x - self.a) / self.eps
        phiR = (x - self.b) / self.eps
        hr = Block(self.hidden)(x)
        hL = Block(self.hidden)(phiL)
        hR = Block(self.hidden)(phiR)
        h  = jnp.concatenate([hr, hL, hR], axis=-1)
        return nn.Dense(1)(h)

def weight_1d(x, a=0.0, b=1.0):
    return jnp.minimum(x - a, b - x)**2

net = BLPINN1D()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 1)))

def u_apply(params, x): return net.apply(params, x).squeeze(-1)

def loss(params, X_int, X_bc, g_bc, f_fn, Leps_fn, w_fn):
    # Leps_fn(params, x) computes the differential operator via nested jax.grad
    res = Leps_fn(params, X_int) - f_fn(X_int)
    Lpde = jnp.mean(w_fn(X_int) * res**2)
    Lbc  = jnp.mean((u_apply(params, X_bc) - g_bc)**2)
    return Lpde + Lbc

def Leps_conv_diff(params, x):                    # -eps u'' + a u' + gamma u
    def u_scalar(xv): return u_apply(params, xv[None]).squeeze()
    du  = jax.vmap(jax.grad(u_scalar))(x)
    d2u = jax.vmap(jax.grad(jax.grad(u_scalar)))(x)
    return -EPS * d2u + A_COEF * du + GAMMA * u_apply(params, x)

# sampling: 500 uniform interior + 500 near each endpoint from N(a, eps^2) truncated
def sample_1d(key, N_int=500, N_layer=500, eps=1e-10):
    k1, k2, k3 = jax.random.split(key, 3)
    xs = [jax.random.uniform(k1, (N_int, 1))]
    for c, k in zip((0.0, 1.0), (k2, k3)):
        z = c + eps * jax.random.normal(k, (N_layer, 1))
        z = jnp.clip(z, 0.0, 1.0)
        xs.append(z)
    return jnp.concatenate(xs, 0)

solver = jaxopt.LBFGS(fun=loss, maxiter=2000, linesearch="zoom",
                      tol=1e-15)                      # paper uses Levenberg-Marquardt; L-BFGS is a close drop-in
state = solver.init_state(params, X_int, X_bc, g_bc, f, Leps_conv_diff, weight_1d)
for _ in range(2000):
    params, state = solver.update(params, state, X_int, X_bc, g_bc, f, Leps_conv_diff, weight_1d)
```

Hyperparameters: single hidden layer, sigmoid activations (smooth derivatives), 50 (1-D) or 35-50 (2-D) neurons per block, 1500 (1-D) / 2500 (2-D) collocation points (1/3 uniform interior + 2/3 truncated-normal `N(boundary, eps^2)`), Levenberg-Marquardt optimizer until loss < 1e-15 or 2000 iters. Boundary-layer locations are detected automatically because each affine `W (phi/eps) + b = 0` is an `O(eps)` shift of the level set.

## Results
1-D convection-diffusion-reaction at `eps = 1e-10`: rel-L2 / Linf ~ 1.4e-8 / 1.8e-8 (Table 1; flat across `eps` from 1e-2 to 1e-10). 1-D reaction-diffusion: 6.5e-8 / 3.2e-8. 1-D coupled system: 2e-8 to 1e-7. 2-D constant-coefficient and variable-coefficient convection-diffusion-reaction: 4-7e-7. Irregular-domain 2-D cases reach 1e-7 with the level-set ansatz. Loss converges to ~1e-15.
