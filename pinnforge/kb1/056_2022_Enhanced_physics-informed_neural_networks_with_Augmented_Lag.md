---
slot: 056
title: "Enhanced physics-informed neural networks with Augmented Lagrangian relaxation method (AL-PINNs)"
authors: [Hwijae Son, Sung Woong Cho, Hyung Ju Hwang]
year: 2022
venue: Neurocomputing (arXiv:2205.01059)
gitrepo: "https://github.com/HwijaeSon/AL-PINNs"
---

## TL;DR
Treat boundary/initial conditions as **hard constraints** and the PDE residual as the objective; relax the constraints with an **Augmented Lagrangian** that adds both a quadratic penalty `beta` and a learnable, point-wise multiplier `lambda(x_b)`. Solving the resulting max-min by gradient descent-ascent gives an adaptive, theoretically convergent loss-balancing scheme that outperforms LR-annealing, NTK, and SA-PINN.

## Problem
Standard PINN loss `L_PDE + L_BC` lumps the BC as a quadratic penalty whose weight must be tuned by hand or by heuristic gradient/NTK rules. A pure penalty needs `beta -> inf` (numerically unstable); a pure Lagrange dual requires local convexity. Boundary loss dominates training and BC violations rarely vanish, so the network never satisfies the boundary exactly.

## Method
Formulate PINN training as
$$
\min_\theta \;\|\mathcal N u_\theta - f\|^2_{L^2(\Omega)} \;\text{ s.t. } \mathcal T u_\theta = g \text{ on } \partial\Omega
$$
and use the Augmented Lagrangian
$$
\mathcal L_\lambda(\theta) = \|\mathcal N u_\theta - f\|^2_{L^2(\Omega)} + \beta\|\mathcal T u_\theta - g\|^2_{L^2(\partial\Omega)} + \langle\lambda,\,\mathcal T u_\theta - g\rangle_{L^2(\partial\Omega)}
$$
Discretize `lambda` as one scalar per boundary collocation point `lambda_j = lambda(x_b^j)` (initial `lambda = 0`). Solve `max_lambda min_theta L_lambda` by gradient descent-ascent:

$$
\theta \leftarrow \theta - \eta_\theta\,\nabla_\theta \mathcal L_\lambda,\qquad
\lambda_j \leftarrow \lambda_j + \eta_\lambda\,(\mathcal T u_\theta(x_b^j) - g(x_b^j))
$$

Recommended ranges (from the paper's sweep): `beta` in {10, 50, 100, 500, 1000}, `eta_lambda` in {1, 0.1, 0.01, 0.001}, `eta_theta` in {1e-3, 1e-4}, Adam, 50k-100k iters. Multiple constraint groups (IC + BC for time-dependent PDEs) get their own `lambda` vectors `lambda^1, lambda^2`. Convergence to the true PDE solution is proved (Gamma'-convergence) for Helmholtz, viscous Burgers, and Klein-Gordon.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    width: int = 50
    depth: int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.width)(x))
        return nn.Dense(1)(x)

beta   = 100.0
eta_l  = 0.1                                  # ascent step on lambda
optimizer = optax.adam(1e-3)

# lam: explicit array; not part of `params`, updated by hand
def init_state(params, n_b):
    return optimizer.init(params), jnp.zeros((n_b, 1))

def bc_residual(params, apply_fn, x_bc, g_b):
    u_b = jax.vmap(apply_fn, in_axes=(None, 0))(params, x_bc)
    return u_b - g_b                          # T u - g

@jax.jit
def train_step(params, opt_state, lam, x_r, x_bc, g_b, apply_fn):
    def loss(p):
        # PDE residual on x_r
        r = jax.vmap(lambda x: pde_operator(p, apply_fn, x) - f(x))(x_r)
        L_pde = jnp.mean(r**2)
        c = bc_residual(p, apply_fn, x_bc, g_b)
        L_pen = beta * jnp.mean(c**2)
        L_lag = jnp.mean(lam * c)
        return L_pde + L_pen + L_lag
    grads = jax.grad(loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    # explicit ascent step on multipliers (stop_gradient on net)
    c_now = jax.lax.stop_gradient(bc_residual(params, apply_fn, x_bc, g_b))
    lam = lam + eta_l * c_now
    return params, opt_state, lam
```

For an IBVP, split the constraint set `{IC, BC_dirichlet, BC_neumann}` and maintain one `lambda` array per set.

## Results
On 2-D Helmholtz, viscous Burgers, and Klein-Gordon, AL-PINNs reach 5-50x lower relative L2 than vanilla PINN, LR-annealing (Wang et al. 2021), NTK-PINN, and SA-PINN at matched compute. Boundary residual drops by 1-3 orders of magnitude versus pure penalty; performance is robust to `beta` over two orders of magnitude. Convergence to the classical solution is proved in Section 3.
