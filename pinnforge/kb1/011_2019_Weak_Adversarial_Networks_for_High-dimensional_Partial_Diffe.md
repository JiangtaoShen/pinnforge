---
slot: 11
title: "Weak Adversarial Networks for High-dimensional Partial Differential Equations"
authors: [Yaohua Zang, Gang Bao, Xiaojing Ye, Haomin Zhou]
year: 2019
venue: "Journal of Computational Physics (arXiv:1907.08272)"
gitrepo: ""
---

## TL;DR
Cast the weak form of a PDE as a min-max (saddle-point) problem: the *primal* network `u_theta` solves the PDE in weak sense; the *adversarial* test-function network `phi_eta` tries to maximise the operator-norm residual `|<A[u_theta], phi_eta>| / ||phi_eta||`. Alternate gradient descent (primal) / ascent (adversarial), GAN-style. Handles high-d, irregular domains and solutions without classical regularity.

## Problem
PINNs / VPINNs need a *fixed* set of test functions (collocation Dirac or polynomial basis); choice biases the loss. In high-d, classical bases blow up combinatorially. Weak solutions may be the only ones that exist (singular forcings, irregular domains).

## Method
For `A[u] = -div(A grad u) + b.grad u + c u - f` on `Omega`, with test `phi in H_0^1`, weak form: `<A[u], phi> = int_Omega [(A grad u).(grad phi) + (b.grad u + c u - f) phi] dx = 0` for all `phi`. Operator norm: `||A[u]||_op = max_{phi} |<A[u],phi>| / ||phi||_2`. Hence `u` is a weak solution iff
$$
u^{*} = \arg\min_{u\in H^1,\,B[u]=0}\;\max_{\phi\in H_0^1}\;\frac{|\langle A[u],\phi\rangle|^2}{\|\phi\|_2^2}
$$
Parameterise: `u_theta`, `phi_eta` are two MLPs. To make `phi_eta = 0` on `dOmega` automatically, factorise `phi_eta(x) = w(x) v_eta(x)` where `w` is a pre-computed signed-distance / smooth cut-off vanishing on `dOmega`. Total loss:
$$
\mathcal{L}(\theta, \eta) = \log\!\bigl|\langle A[u_\theta], \phi_\eta\rangle_{N_r}\bigr|^2 - \log\|\phi_\eta\|_{2,N_r}^{2} + \alpha\,\mathcal{L}_{bdry}(\theta)
$$
$$
\mathcal{L}_{bdry}(\theta) = \tfrac{1}{N_b}\sum_j |u_\theta(x_b^j) - g(x_b^j)|^2 \;(\text{Dirichlet})
$$
Inner products are Monte-Carlo approximated over `N_r` uniform samples in `Omega`. Alternate updates:
$$
\theta \leftarrow \theta - \tau_\theta\,\nabla_\theta\mathcal{L},\qquad \eta \leftarrow \eta + \tau_\eta\,\nabla_\eta\mathcal{L}
$$
with `K_u` primal steps per `K_phi` adversarial steps (e.g. `K_u : K_phi = 2 : 1`).

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    h: int = 20
    depth: int = 6
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.h)(x))
        return nn.Dense(1)(x)

u_net, phi_net = MLP(), MLP()
k1, k2 = jax.random.split(jax.random.PRNGKey(0))
u_params   = u_net.init  (k1, jnp.zeros((1, d)))
phi_params = phi_net.init(k2, jnp.zeros((1, d)))
opt_u, opt_phi = optax.adam(1.5e-3), optax.adam(4e-3)
su, sp = opt_u.init(u_params), opt_phi.init(phi_params)

def w_cutoff(x):                                       # ~ vanishes on dOmega
    return jnp.prod(x * (1 - x), axis=1, keepdims=True)   # e.g. unit hypercube

def weak_residual(u_p, phi_p, x_int, A_mat, b_vec, c_fn, f_fn):
    def u_scalar(p, xi):   return u_net.apply  (p, xi[None])[0, 0]
    def phi_scalar(p, xi): return (w_cutoff(xi[None]) * phi_net.apply(p, xi[None]))[0, 0]
    u   = jax.vmap(lambda xi: u_scalar  (u_p,   xi))(x_int)
    phi = jax.vmap(lambda xi: phi_scalar(phi_p, xi))(x_int)
    gu  = jax.vmap(lambda xi: jax.grad(u_scalar,   argnums=1)(u_p,   xi))(x_int)
    gp  = jax.vmap(lambda xi: jax.grad(phi_scalar, argnums=1)(phi_p, xi))(x_int)
    Au_phi = ((gu @ A_mat) * gp).sum(1) + (b_vec * gu).sum(1) * phi \
             + c_fn(x_int) * u * phi - f_fn(x_int) * phi
    inner = Au_phi.mean()
    return inner**2 / (jnp.mean(phi**2) + 1e-12)

alpha, K_u, K_phi = 1e4, 2, 1

def loss_u(u_p, phi_p, x_int, x_bc, g_bc):
    return weak_residual(u_p, phi_p, x_int, A_mat, b_vec, c_fn, f_fn) \
           + alpha * jnp.mean((u_net.apply(u_p, x_bc) - g_bc) ** 2)

def loss_phi(phi_p, u_p, x_int):
    return -weak_residual(u_p, phi_p, x_int, A_mat, b_vec, c_fn, f_fn)

@jax.jit
def step_u(u_p, phi_p, su, x_int, x_bc, g_bc):
    g = jax.grad(loss_u)(u_p, phi_p, x_int, x_bc, g_bc)
    upd, su = opt_u.update(g, su, u_p)
    return optax.apply_updates(u_p, upd), su

@jax.jit
def step_phi(phi_p, u_p, sp, x_int):
    g = jax.grad(loss_phi)(phi_p, u_p, x_int)
    upd, sp = opt_phi.update(g, sp, phi_p)
    return optax.apply_updates(phi_p, upd), sp
```

For parabolic PDEs, lift time `t` as an input coordinate and add IC penalty `int_Omega |u(x,0) - h(x)|^2 dx`.

Recommended: width 20-40, depth 4-8 for both networks; `K_u : K_phi = 2 : 1` or `1 : 1`; Adam `lr_u ~ 1.5e-3`, `lr_phi ~ 4e-3`; `alpha = 1e3-1e4`; `N_r = 4000` interior MC points (independent of dimension).

## Results
On 2-D, 5-D, 10-D Poisson with irregular domains and on parabolic PDEs in d=5..20, WAN matches reference (FEM in 2-D, mesh-free Galerkin elsewhere) within 1-3% relative L2; outperforms strong-form PINN/DGM on solutions with low regularity. Cost grows linearly (not exponentially) with d.
