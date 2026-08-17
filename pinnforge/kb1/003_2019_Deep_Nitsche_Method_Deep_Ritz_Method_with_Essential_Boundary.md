---
slot: 3
title: "Deep Nitsche Method: Deep Ritz Method with Essential Boundary Conditions"
authors: [Yulei Liao, Pingbing Ming]
year: 2019
venue: "Communications in Computational Physics (arXiv:1912.01309)"
gitrepo: ""
---

## TL;DR
Augment the Deep Ritz energy functional with Nitsche's surface terms so that essential (Dirichlet) boundary conditions are imposed weakly but *consistently* (no soft-penalty bias). The resulting loss has one extra penalty parameter `beta` and remains unconstrained, training proceeds with SGD/Adam exactly as Deep Ritz.

## Problem
Deep Ritz minimises the energy but cannot enforce Dirichlet BCs since DNN trial functions are non-interpolatory. The naive quadratic-penalty approach is *inconsistent* (gives sub-optimal rates and biased solutions); exact-BC constructions (distance function * NN) are hard for complex geometries; Lagrange multipliers add a constrained min-max problem.

## Method
Mixed BVP: `-div(A grad u) = f` on `Omega`, `u = g_D` on `Gamma_D`, `(A grad u).n = g_N` on `Gamma_N`. The Deep Nitsche energy functional (over a DNN ansatz `u_n in H_n`) is:

$$
I[v] = \tfrac{1}{2}\!\int_\Omega A\nabla v\!\cdot\!\nabla v\,dx
 - \int_\Omega f v\,dx
 - \int_{\Gamma_N}\! g_N v\,d\sigma
 + \int_{\Gamma_D}(g_D - v)\,\partial_\nu v\,d\sigma
 + \tfrac{\beta}{2}\!\int_{\Gamma_D}(g_D - v)^2 d\sigma
$$

The Dirichlet surface integrals replace the inconsistent penalty `(beta/2) int (v - g_D)^2`. The conormal-derivative cross term `(g_D - v) d_nu v` restores *consistency*: the minimiser of `I` over the true solution space is the exact `u` regardless of `beta`. The quadratic term ensures coercivity; pick `beta = O(1/h) ~ O(N^(1/d))` or a fixed moderate value (paper uses `beta in {100, 500}`).

Discretise integrals by Monte-Carlo / quasi-Monte-Carlo sampling, then minimise by Adam.

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class ResBlock(nn.Module):
    h: int
    @nn.compact
    def __call__(self, x):
        y = jnp.tanh(nn.Dense(self.h)(x))
        y = jnp.tanh(nn.Dense(self.h)(y))
        return x + y

class DeepRitzNet(nn.Module):
    h: int = 10
    blocks: int = 5
    out_dim: int = 1
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.h)(x)
        for _ in range(self.blocks):
            x = ResBlock(self.h)(x)
        return nn.Dense(self.out_dim)(x)

net = DeepRitzNet()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
beta = 500.0
A = jnp.eye(2)

def u_apply(params, x):  return net.apply(params, x)

def grad_u(params, x):
    def u_single(p, xi):  return u_apply(p, xi[None])[0, 0]
    return jax.vmap(lambda xi: jax.grad(u_single, argnums=1)(params, xi))(x)

def nitsche_loss(params, x_int, x_D, x_N, normals_D, f, gD, gN):
    u_int = u_apply(params, x_int)[:, 0]
    g_int = grad_u(params, x_int)                          # (N, 2)
    bulk = 0.5 * jnp.mean((g_int @ A * g_int).sum(axis=1)) - jnp.mean(f * u_int)
    neu  = -jnp.mean(gN * u_apply(params, x_N)[:, 0])

    uD = u_apply(params, x_D)[:, 0]
    gD_grad = grad_u(params, x_D)
    dnu_v = (gD_grad * normals_D).sum(axis=1)
    cross = jnp.mean((gD - uD) * dnu_v)
    pen   = 0.5 * beta * jnp.mean((gD - uD)**2)
    return bulk + neu + cross + pen

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    grads = jax.grad(nitsche_loss)(params, *batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

Recommended (paper): 5 residual blocks of width 10 (~1141 params) for 2-D, scaled up for high-D; Adam `lr=1e-3`, 50k epochs; QMC Halton sampling with 512 interior + 64 per face.

## Results
On 2-D mixed BVPs (smooth and singular solutions) relative L2/H1 errors of 1e-2 to 1e-3 with ~1000 trainable parameters; scales to 100-D Dirichlet problems with 10^4-10^5 parameters at similar accuracy. Outperforms quadratic-penalty Deep Ritz especially on boundary error.
