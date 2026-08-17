---
slot: 73
title: "An analysis and solution of ill-conditioning in physics-informed neural networks"
authors: [Wenbo Cao, Weiwei Zhang]
year: 2024
venue: "Journal of Computational Physics (arXiv:2405.01957)"
gitrepo: ""
---

## TL;DR
The training failure of PINNs on stiff/high-Re problems is traced to the ill-conditioning of the *Jacobian of the PDE system itself*, not the neural network Hessian. The authors construct a **controlled system** that re-parameterises the steady solution as a sequence of well-conditioned sub-problems (a pseudo-time stepping scheme, TSONN), enabling the first PINN simulation of 3-D laminar flow over the ONERA-M6 wing at Re=5,000.

## Problem
For PDE `f(q)=0` with steady solution `q_s`, the convergence rate of any iterative solver depends on `kappa(J(q_s))`, where `J = df/dq`. For lid-driven cavity at Re=2,500 the FDM Jacobian has `kappa ~ 1.3e4` and vanilla PINNs fail entirely. Loss-balancing alone cannot fix this because the conditioning lives in the operator, not in the relative weights.

## Method
Add a **linear forcing term** that shifts all Jacobian eigenvalues left while preserving the steady solution:
$$ f_c(q) = f(q) - \mu (q - q_s) = 0, \quad \mu > 0 $$
Eigenvalues become `lambda_i - mu`, so kappa decreases as mu grows. Since `q_s` is unknown, substitute the current network output `q_n = u(.;theta_n)` and iterate (outer loop). Equivalently this is implicit pseudo-time stepping with `Delta_t = 1/mu`:
$$ (q - q_n)/\Delta t = f(q) $$

Apply the forcing to **every** operator (PDE, BC, IC). For Dirichlet BCs / ICs use the known target directly in place of `q_s`. Choose the sign of `f` so all Jacobian eigenvalues are negative (e.g. Burgers: `g(q) = -q_t - q q_x + q_xx`; Dirichlet BC: `h(q) = c - q`).

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from jax.flatten_util import ravel_pytree

class PINN(nn.Module):                              # MLP, tanh, 5x128 or 8x128
    H: int = 128
    depth: int = 5
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.H)(x))
        return nn.Dense(1)(x)

def u_fn(params, x):
    return PINN().apply(params, x)

def pde_residual(params, x):                        # f(q) with stable sign convention
    u   = lambda xi: u_fn(params, xi).squeeze()
    u_t = lambda xi: jax.grad(u)(xi)[1]             # t = x[1]
    u_x = lambda xi: jax.grad(u)(xi)[0]
    u_xx= lambda xi: jax.grad(u_x)(xi)[0]
    return jax.vmap(lambda xi: -u_t(xi) - u(xi)*u_x(xi) + u_xx(xi))(x)  # Burgers

def bc_residual(params, x_bc, c):                   # h(q) = c - q
    return c - u_fn(params, x_bc)

optimizer = optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=20))
opt_state = optimizer.init(params)

@jax.jit
def loss_fn(params, coloc, x_bc, c_bc, q_n, mu, w_pde, w_bc):
    r_pde = pde_residual(params, coloc) - mu*(u_fn(params, coloc).squeeze() - q_n)
    r_bc  = bc_residual(params, x_bc, c_bc)
    return w_pde*jnp.mean(r_pde**2) + w_bc*jnp.mean(r_bc**2)

prev_params = params
for outer in range(N):
    coloc = sample(domain)                          # random resample per outer step
    q_n   = jax.lax.stop_gradient(u_fn(params, coloc).squeeze())
    try:
        for inner in range(K):                      # L-BFGS inner iterations
            grads = jax.grad(loss_fn)(params, coloc, x_bc, c_bc, q_n, mu, w_pde, w_bc)
            updates, opt_state = optimizer.update(
                grads, opt_state, params,
                value=loss_fn(params, coloc, x_bc, c_bc, q_n, mu, w_pde, w_bc),
                grad=grads, value_fn=lambda p: loss_fn(p, coloc, x_bc, c_bc, q_n, mu, w_pde, w_bc))
            params = optax.apply_updates(params, updates)
        prev_params = params
    except FloatingPointError:
        params = prev_params                        # NaN guard
```

Hyperparameters: tanh MLP 5-8 layers x 128 units; L-BFGS with strong Wolfe (zoom line-search in optax); outer iterations `N ~ 1e3`, inner `K ~ 50-200`; pseudo-time step `Delta_t = 0.3` (i.e. `mu ~ 3`) for airfoil/wing; volume-weighted residuals for non-uniform collocation; resample every outer step. PDE form must give negative-eigenvalue Jacobian; volume- and relative-weighting are *inside* the residual, pseudo-time stepping is the outermost layer.

## Results
TSONN solves Re=5,000 flow over NACA0012 (rel-L2 ~2%) and over the 3-D M6 wing (6% wall pressure error), both unreachable by vanilla PINNs. Vanilla PINNs are recovered as `Delta_t -> inf` (mu -> 0); smaller `Delta_t` gives more well-conditioned sub-problems but needs more outer steps.
