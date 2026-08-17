---
slot: 2
title: "An Energy Approach to the Solution of Partial Differential Equations in Computational Mechanics via Machine Learning: Concepts, Implementation and Applications"
authors: [E. Samaniego, C. Anitescu, S. Goswami, V. M. Nguyen-Thanh, T. Rabczuk]
year: 2019
venue: "Computer Methods in Applied Mechanics and Engineering (arXiv:1908.10407)"
gitrepo: ""
---

## TL;DR
Deep Energy Method (DEM): replace the residual-MSE loss of PINNs by the total potential energy of the mechanical system, integrated over the domain by numerical quadrature, and enforce Dirichlet BCs *exactly* by hard-coding them into the network output. The PDE order is effectively halved (energy uses first-order grads only), training is faster and more stable than residual-form PINN for elasticity / hyperelasticity problems.

## Problem
Strong-form PINN minimises mean-square residual of the PDE, requiring second-order autograd derivatives and a soft penalty for boundary conditions. For solid mechanics this is unnecessarily expensive: the variational/energy form is naturally available, contains only first-order derivatives, and Neumann BCs appear naturally.

## Method
Let `u_p(x)` be a DNN with parameters `p`. For a linear elastic body with strain `eps(u) = (grad u + grad u^T)/2`, elasticity tensor `C`, stored strain energy `Psi(eps) = (1/2) eps : C : eps`, applied traction `t_bar` on Neumann boundary `Gamma_N`, and body force `b`, define the total potential energy:
$$
E[u] = \int_\Omega \Psi(\varepsilon(u))\,d\Omega - \int_\Omega b\cdot u\,d\Omega - \int_{\Gamma_N} \bar{t}\cdot u\,d\Gamma
$$
Approximate the integrals by quadrature (uniform sampling with weights `w_i`):
$$
\mathcal{L}(p) = \sum_i \Psi(\varepsilon(u_p(x_i)))\,w_i \;-\; \sum_i b\cdot u_p(x_i) w_i \;-\; \sum_j \bar{t}\cdot u_p(x_j^{\partial}) w_j^{\partial}
$$
Hard Dirichlet BCs: parameterise the trial function so they are exact by construction, e.g. on the quarter-cylinder with `u(0,y)=0`, `v(x,0)=0`:
$$
u(x,y) = x \cdot \hat u_{NN}(x,y), \qquad v(x,y) = y \cdot \hat v_{NN}(x,y)
$$
Then `L` contains NO boundary-error term — only the energy.

For hyperelasticity (neo-Hookean), replace `Psi` by `(mu/2)(I_1 - 3) - mu ln J + (lam/2)(ln J)^2` where `F = I + grad u`, `J = det F`, `I_1 = tr(F^T F)`. Same loss machinery.

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    hidden: int = 30
    depth: int = 4
    out_dim: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(self.out_dim)(x)

net = MLP()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))

def trial(params, xy):                              # hard Dirichlet
    raw = net.apply(params, xy)
    x, y = xy[:, 0:1], xy[:, 1:2]
    u = x * raw[:, 0:1]
    v = y * raw[:, 1:2]
    return jnp.concatenate([u, v], axis=1)

def strain_energy(params, xy, lam, mu):
    # Per-point Jacobian of displacement field via jax.jacrev.
    def u_single(p, xi):  return trial(p, xi[None])[0]   # (2,)
    J = jax.vmap(lambda xi: jax.jacrev(u_single, argnums=1)(params, xi))(xy)
    du_dx, du_dy = J[:, 0, 0], J[:, 0, 1]
    dv_dx, dv_dy = J[:, 1, 0], J[:, 1, 1]
    eps_xx, eps_yy = du_dx, dv_dy
    eps_xy = 0.5 * (du_dy + dv_dx)
    tr = eps_xx + eps_yy
    return 0.5 * lam * tr**2 + mu * (eps_xx**2 + eps_yy**2 + 2 * eps_xy**2)

def loss_fn(params, xy_int, w_int, xy_neu, w_neu, t_bar, lam, mu):
    L_int = jnp.sum(strain_energy(params, xy_int, lam, mu) * w_int)
    L_neu = -jnp.sum((t_bar * trial(params, xy_neu)).sum(axis=1) * w_neu)
    return L_int + L_neu

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    grads = jax.grad(loss_fn)(params, *batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

Recommended: Adam `lr = 1e-3`; later switch to L-BFGS; Xavier init; tanh activation; depth 3-4; width 20-50. Quadrature: uniform grid with trapezoidal weights, or Gauss points if IGA-style.

## Results
DEM reproduces analytical solutions of thick-cylinder under pressure, plate with hole, hollow sphere, and cube with spherical hole to relative L2 errors ~1e-3, matching or beating collocation PINN with shallower networks and ~halved training time (no second-derivative autograd).
