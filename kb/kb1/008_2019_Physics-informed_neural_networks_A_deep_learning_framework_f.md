---
slot: 8
title: "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"
authors: [M. Raissi, P. Perdikaris, G. Karniadakis]
year: 2019
venue: "Journal of Computational Physics"
gitrepo: ""
doi: "10.1016/j.jcp.2018.10.045"
---

## TL;DR
The foundational PINN paper. Approximate the PDE solution `u(t,x)` by a deep MLP `u_theta`, define a second network `f_theta = u_t + N[u]` via automatic differentiation (sharing parameters), and train by minimising MSE of `f` at random collocation points + MSE of `u` at supervised points (IC/BC, or sparse measurements). Two variants are introduced: *continuous-time* (autograd everywhere in space-time) and *discrete-time* (implicit Runge-Kutta with q stages, networks predict the q internal stages).

## Problem
Traditional solvers fail for inverse problems (parameter identification from sparse data), unknown-PDE discovery, and high-dimensional / mesh-free settings. Pure deep learning is data-hungry. The remedy: bake the PDE residual into the loss via autograd as both regulariser and "discretisation".

## Method

### A. Continuous-time PINN
Approximate `u(t,x) ~ u_theta(t,x)` by an MLP (tanh activations, no regularisation). Build the physics network
$$
f(t,x) = u_t + N[u;\lambda]
$$
by composing `u_theta` with the differential operator using autograd (chain rule). Loss:
$$
\mathcal{L} = \mathrm{MSE}_u + \mathrm{MSE}_f, \quad
\mathrm{MSE}_u = \tfrac{1}{N_u}\!\sum_i |u_\theta(t^i_u, x^i_u)-u^i|^2,\quad
\mathrm{MSE}_f = \tfrac{1}{N_f}\!\sum_i |f(t^i_f, x^i_f)|^2
$$
For forward problems `lambda` is fixed and the `u` term carries IC/BC; for inverse problems `lambda` is trainable and the `u` term carries scattered measurements. Optimise with L-BFGS (full batch) for small data, Adam for large.

### B. Discrete-time PINN (multi-step in one jump)
Apply a q-stage implicit Runge-Kutta to `u_t = -N[u]` between `t^n` and `t^{n+1} = t^n + dt`:
$$
u^{n+c_j}(x) = u^n(x) - \Delta t\sum_{i=1}^{q} a_{ji}\,N[u^{n+c_i}(x)],\quad
u^{n+1}(x) = u^n(x) - \Delta t\sum_{i=1}^{q} b_i\,N[u^{n+c_i}(x)]
$$
Parameterise the network to output the q stage values at once: `[u^{n+c_1}, ..., u^{n+c_q}, u^{n+1}](x) = NN(x; theta)`. Match these to the known data at `t^n` via the IRK equations rearranged as `u^n = u^{n+c_j} + dt * sum a_ji N[u^{n+c_i}]`. Train so MSE of the inferred `u^n` against measurements is small. Allows huge `dt` (e.g. `dt = 0.8` for Allen-Cahn) with truncation error `O(dt^{2q})`.

JAX (continuous-time, forward problem):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class PINN(nn.Module):
    hidden: int = 20
    depth: int = 8
    @nn.compact
    def __call__(self, tx):
        # Dense layers with Xavier init (flax default = Glorot uniform).
        for _ in range(self.depth):
            tx = jnp.tanh(nn.Dense(self.hidden)(tx))
        return nn.Dense(1)(tx)

net = PINN()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))

def u_apply(params, tx):  return net.apply(params, tx)[:, 0]

def pde_residual(params, tx, lam):     # Burgers: u_t + lam0 u u_x - lam1 u_xx
    def u_single(p, ti, xi):
        return u_apply(p, jnp.array([[ti, xi]]))[0]
    u   = jax.vmap(lambda t, x: u_single(params, t, x))(tx[:, 0], tx[:, 1])
    u_t = jax.vmap(lambda t, x: jax.grad(u_single, argnums=1)(params, t, x))(tx[:, 0], tx[:, 1])
    u_x = jax.vmap(lambda t, x: jax.grad(u_single, argnums=2)(params, t, x))(tx[:, 0], tx[:, 1])
    u_xx = jax.vmap(lambda t, x: jax.grad(jax.grad(u_single, argnums=2), argnums=2)(params, t, x))(tx[:, 0], tx[:, 1])
    return u_t + lam[0] * u * u_x - lam[1] * u_xx

def loss_fn(params, tx_u, u_target, tx_f, lam):
    L_u = jnp.mean((u_apply(params, tx_u) - u_target) ** 2)
    L_f = jnp.mean(pde_residual(params, tx_f, lam) ** 2)
    return L_u + L_f

lam = jnp.array([1.0, 0.01 / jnp.pi])     # fixed for forward problem
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, tx_u, u_target, tx_f):
    grads = jax.grad(loss_fn)(params, tx_u, u_target, tx_f, lam)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
# For L-BFGS use jaxopt.LBFGS or optax.scale_by_lbfgs.
```

Recommended (paper): tanh activations, Xavier init, depth 4-9, width 20-200; L-BFGS for forward problems with `Nu ~ 100`, `Nf ~ 10000`; Adam for large datasets / inverse. For discrete-time, q=50-500 stages routinely work (use Butcher tableau from Gauss-Legendre).

Inverse / discovery: add `lambda` to the parameter pytree (e.g. `{"net": net_params, "lam": jnp.array([...])}`); it co-trains with weights.

## Results
On Schrodinger, Allen-Cahn, KdV, Burgers, Navier-Stokes (cylinder wake), relative L2 errors `1e-3` to `1e-5` with ~100 measurement points. Inverse: identifies Burgers viscosity, Navier-Stokes Re, KdV coefficients with <1% error from sparse noisy measurements. Discrete-time PINN takes a *single* dt=0.8 step from t=0.1 to t=0.9 for Allen-Cahn with 4 layers x 200 neurons.
