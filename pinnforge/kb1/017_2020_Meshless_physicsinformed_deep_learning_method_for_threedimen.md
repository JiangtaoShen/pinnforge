---
slot: 17
title: "Meshless physics-informed deep learning method for three-dimensional solid mechanics"
authors: [D. Abueidda, Q. Lu, S. Koric]
year: 2020
venue: "International Journal for Numerical Methods in Engineering (arXiv:2012.01547)"
gitrepo: ""
---

## TL;DR
Application paper: vanilla strong-form PINN ("Deep Collocation Method", DCM) for 3-D solid mechanics with multiple constitutive laws — linear elasticity, neo-Hookean hyperelasticity at large deformation, and J2 (von Mises) plasticity with isotropic *and* kinematic hardening. The only architectural element is a plain MLP that outputs the displacement field `u_hat(X)` from coordinates `X`; strains, stresses, and constitutive integrations are computed symbolically via autograd and finite-step plastic return mapping.

## Problem
3-D nonlinear solid mechanics with FEM requires meshing, tangent-modulus assembly and Newton-Raphson with linear-system solves. DEM works for problems with energy potentials but plasticity does not have a clean variational form. Need a meshless approach that handles arbitrary constitutive laws.

## Method
Network: dense MLP, 6 layers `[3, 60, 60, 60, 60, 3]`, hyperbolic-tangent (tanh) activations, mapping coordinates `X = (x,y,z)` to displacement `u_hat = (u_x, u_y, u_z)`.

Loss (quasi-static, no time):
$$
\mathcal{L} = \mathrm{MSE}_G + \lambda_u\,\mathrm{MSE}_u + \lambda_t\,\mathrm{MSE}_t
$$
- `MSE_G = (1/N_G) sum |div(sigma(u_hat))|^2` — strong-form equilibrium at `N_G` interior collocation points (sampled by Monte-Carlo).
- `MSE_u = (1/N_u) sum |u_hat - u_bar|^2` on essential boundary `Gamma_u`.
- `MSE_t = (1/N_t) sum |t_hat - t_bar|^2` on natural boundary `Gamma_t`.

Stresses depend on the constitutive law:
- **Linear elasticity**: `sigma = lam tr(eps) I + 2 mu eps`, `eps = (grad u + grad u^T)/2`.
- **Neo-Hookean (large deformation)**: `F = I + grad u`, `J = det F`, `b = F F^T`, `sigma = (mu/J)(b - I) + (lam/J) ln(J) I`.
- **J2 plasticity (isotropic + kinematic hardening)**: at each collocation point, compute trial elastic stress `sigma_trial`, evaluate yield function `F = ||s_trial - alpha|| - sqrt(2/3)(sigma_Y + H_iso e_p)`. If `F > 0` perform return mapping (closed-form for linear hardening) to update plastic strain `e_p`, back-stress `alpha`, and corrected stress. `e_p, alpha` are *internal variables* stored per collocation point and updated each load step.

Loading is incremented over pseudo-time steps; at each step the network is (re-)trained with current `u_bar`, `t_bar`.

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class DCM(nn.Module):
    @nn.compact
    def __call__(self, X):
        for _ in range(4):
            X = jnp.tanh(nn.Dense(60)(X))
        return nn.Dense(3)(X)

net = DCM()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))

def u_apply(params, X): return net.apply(params, X)

def deformation_gradient(params, X):
    # Per-point Jacobian of displacement w.r.t. coordinates (N, 3, 3).
    def u_single(p, Xi):  return u_apply(p, Xi[None])[0]
    J_u = jax.vmap(lambda Xi: jax.jacrev(u_single, argnums=1)(params, Xi))(X)
    I3 = jnp.eye(3)
    return I3 + J_u                                          # (N, 3, 3)

def neo_hookean_stress(F, lam, mu):
    J = jnp.linalg.det(F)[:, None, None]
    b = F @ jnp.swapaxes(F, -1, -2)
    I = jnp.broadcast_to(jnp.eye(3), b.shape)
    return (mu / J) * (b - I) + (lam / J) * jnp.log(J) * I

def divergence(params, X, lam, mu):
    # divergence of sigma at each X: compute via outer jax.jacrev of stress wrt X.
    def stress_single(p, Xi):
        F = deformation_gradient(p, Xi[None])
        return neo_hookean_stress(F, lam, mu)[0]             # (3, 3)
    dsig = jax.vmap(lambda Xi: jax.jacrev(stress_single, argnums=1)(params, Xi))(X)
    return jnp.einsum("nijj->ni", dsig)                       # (N, 3)

def loss_fn(params, X_int, X_u, u_bar, X_t, t_bar, n_t, lam, mu):
    div_sig = divergence(params, X_int, lam, mu)
    L_G = jnp.mean(jnp.sum(div_sig ** 2, axis=1))
    L_u = jnp.mean(jnp.sum((u_apply(params, X_u) - u_bar) ** 2, axis=1))
    F_b   = deformation_gradient(params, X_t)
    sig_b = neo_hookean_stress(F_b, lam, mu)
    t_pred = jnp.einsum("nij,nj->ni", sig_b, n_t)
    L_t = jnp.mean(jnp.sum((t_pred - t_bar) ** 2, axis=1))
    return L_G + lam_u * L_u + lam_t * L_t

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    grads = jax.grad(loss_fn)(params, *batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
# Use jaxopt.LBFGS for the post-Adam refinement.
```

Recommended: depth 4-6, width 60, tanh; `lambda_u ~ 1e3`, `lambda_t ~ 1e2` (tune so terms have similar magnitude); Adam + L-BFGS sequentially; `N_G ~ 5000-10000` Monte-Carlo points; for plasticity, store and update `e_p, alpha` between load increments.

## Results
On 3-D benchmark geometries (cylinder under tension, plate with spherical hole, cantilever beam, hyperelastic block, plasticity bar), DCM matches commercial FEM (Abaqus) displacement and stress fields within 2-5% relative error without any training data, demonstrating PINN feasibility for general 3-D nonlinear solid mechanics.
