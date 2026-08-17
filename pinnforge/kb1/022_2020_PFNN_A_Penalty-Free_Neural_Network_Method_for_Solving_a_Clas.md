---
slot: 022
title: "PFNN: A Penalty-Free Neural Network Method for Solving a Class of Second-Order Boundary-Value Problems on Complex Geometries"
authors: [Hailong Sheng, Chao Yang]
year: 2020
venue: "Journal of Computational Physics (arXiv:2004.06490)"
gitrepo: ""
---

## TL;DR
Solve second-order BVPs with the deep Ritz energy form and a TWO-network ansatz: one network g learns the Dirichlet boundary, another f handles the interior, glued by a hand-built length factor ell(x) so the trial solution exactly satisfies essential BCs without any boundary penalty term.

## Problem
Variational/Ritz PINNs (DeepRitz, DGM) add boundary penalties beta*L_b that conflict with the energy term and degrade accuracy. Earlier hard-BC constructions only work on simple geometries. Need a way to exactly impose Dirichlet BCs on arbitrary 2-D/3-D domains while keeping the variational (weak-form) loss penalty-free.

## Method
Take BVP -div(rho(|grad u|) grad u) + h(u) = 0 with Dirichlet phi on Gamma_D and Neumann psi on Gamma_N. Its Ritz energy is
$$
I[w] = \int_\Omega (P(w) + H(w))\,dx - \int_{\Gamma_N}\psi w\,dx,\quad P(w)=\!\int_0^{|\nabla w|}\!\rho(s)s\,ds
$$

Two networks are composed with a length factor ell(x) (ell = 0 on Gamma_D, ell > 0 elsewhere):
$$
w_\theta(x) = g_{\theta_1}(x) + \ell(x)\,f_{\theta_2}(x)
$$
ell is built from spline functions l_k per boundary segment gamma_k subset Gamma_D, each l_k=0 on gamma_k, l_k=1 on a non-adjacent "companion" segment, then ell(x) = max_k (1 - (1 - l_k(x))^mu). g handles BC, f handles interior — both trained separately.

Two losses (no cross penalty):
$$
\Phi[g_{\theta_1}] = \tfrac{1}{|S(\Gamma_D)|}\sum_{x_i\in S(\Gamma_D)} |\phi(x_i) - g_{\theta_1}(x_i)|^2
$$
$$
\Psi[w_\theta] = \tfrac{|\Omega|}{|S(\Omega)|}\sum_{x_i\in S(\Omega)} P(w_\theta) + H(w_\theta) - \tfrac{|\Gamma_N|}{|S(\Gamma_N)|}\sum_{x_i\in S(\Gamma_N)}\psi w_\theta
$$

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    width: int = 40
    depth: int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.width)(x))
        return nn.Dense(1)(x)

class PFNN(nn.Module):
    width: int = 40
    depth: int = 4
    # ell_fn passed via __call__ — not a flax submodule
    @nn.compact
    def __call__(self, x, ell_fn):
        g = MLP(self.width, self.depth, name="g")(x)
        f = MLP(self.width, self.depth, name="f")(x)
        return g + ell_fn(x) * f

def ritz_loss(params, apply_fn, ell_fn, x_int, x_neu,
              rho, h_antideriv, psi, vol_om, vol_neu):
    def w(x):
        return apply_fn(params, x, ell_fn)[0]
    def integrand(x):
        gw = jax.grad(w)(x)
        gn = jnp.linalg.norm(gw)
        P  = 0.5 * rho(gn) * gn**2   # exact when rho is constant
        H  = h_antideriv(w(x))
        return P + H
    interior = vol_om * jnp.mean(jax.vmap(integrand)(x_int))
    neumann  = vol_neu * jnp.mean(jax.vmap(lambda x: psi(x) * w(x))(x_neu))
    return interior - neumann

# Stage 1: train only the 'g' sub-tree on Gamma_D
def loss_g(params, apply_fn, x_dir, phi):
    g_only = lambda x: apply_fn({"params": {"g": params}}, x)[0]
    return jnp.mean((jax.vmap(g_only)(x_dir) - phi(x_dir))**2)

# Build & init
net = PFNN()
key = jax.random.PRNGKey(0)
params = net.init(key, jnp.zeros(2), lambda x: jnp.array(1.0))
opt = optax.adam(1e-3)
g_state = opt.init(params["params"]["g"])

@jax.jit
def step_g(p_g, st, x_dir):
    gloss = lambda pg: jnp.mean(
        (jax.vmap(lambda x: MLP().apply({"params": pg}, x)[0])(x_dir) - phi(x_dir))**2)
    grads = jax.grad(gloss)(p_g)
    upd, st = opt.update(grads, st, p_g)
    return optax.apply_updates(p_g, upd), st

# Stage 2: freeze g, train only f via ritz_loss
f_state = opt.init(params["params"]["f"])

@jax.jit
def step_f(p_f, p_g, st, x_int, x_neu):
    def loss_fn(pf):
        full = {"params": {"g": p_g, "f": pf}}
        return ritz_loss(full, net.apply, ell_fn, x_int, x_neu,
                         rho, h_antideriv, psi, vol_om, vol_neu)
    grads = jax.grad(loss_fn)(p_f)
    upd, st = opt.update(grads, st, p_f)
    return optax.apply_updates(p_f, upd), st
```

Recommended: tanh, ResNet-style blocks, Adam then L-BFGS, mu=2 in length factor, n>=4 boundary segments.

## Results
On Poisson and p-Laplacian over L-shape, annulus, and complex 2-D/3-D domains, PFNN beats DGM and Deep Ritz (with penalty) by 1-2 orders of magnitude in L2 error at equal parameter count, especially when essential BCs dominate.
