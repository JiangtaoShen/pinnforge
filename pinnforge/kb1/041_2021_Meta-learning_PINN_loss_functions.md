---
slot: 041
title: "Meta-learning PINN loss functions"
authors: [Apostolos F. Psaros, Kenji Kawaguchi, George Em Karniadakis]
year: 2021
venue: Journal of Computational Physics (arXiv:2107.05544)
gitrepo: ""
---

## TL;DR
A bi-level meta-learning scheme discovers a parametric PINN loss `ℓ_η` offline by differentiating the inner PINN-training trajectory and optimising the outer MSE on a family of parametric PDE tasks. The resulting learned loss (FFN or LAL parametrisation) transfers to unseen, even out-of-distribution, tasks and architectures, beating MSE, L1, Cauchy and online adaptive losses.

## Problem
The default MSE loss in PINNs is not necessarily optimal for any given task family. Pick the wrong loss shape and convergence stalls or generalisation suffers. Online adaptive losses add training overhead and ignore task-distribution priors.

## Method
Replace the fixed per-term squared norm with a parametric loss `ℓ_η(prediction, target)`. The PINN inner objective for task `τ` (PDE parameter `λ_τ`) becomes
$$ L_\tau(\theta,\eta) = w_f L_f + w_b L_b + w_{u_0} L_{u_0} \quad\text{with}\quad L_\bullet = \tfrac{1}{N_\bullet}\sum \ell_\eta(\hat{u}_\theta, \text{target}) $$
Bi-level optimisation:
$$ \min_\eta L_O\!\big(\theta^*(\eta),\eta\big),\quad \theta^*_\tau(\eta) = \arg\min_\theta L_\tau(\theta,\eta) $$
where the outer objective is plain MSE on validation points. The meta-gradient is computed by differentiating through `J` inner SGD steps:
$$ \theta_\tau^{(J)} = \theta_\tau^{(0)} - \epsilon_2 \sum_{j=1}^{J} \nabla_\theta L_\tau(\theta^{(j-1)},\eta) $$

Two parametrisations:
- **LAL** (learned adaptive loss): `ρ_{α,c}(d) = (|α-2|/α)·(((d/c)²/|α-2| + 1)^{α/2} - 1)`. Two scalars per dimension; the authors prove it satisfies the optimal-stationarity and MSE-relation conditions automatically.
- **FFN**: a small ReLU MLP `ℓ_η(û,u)` (2 hidden layers × 40, softplus output). Needs the regularizer
$$ L_{O,\text{add}}(\eta) = \mathbb{E}_q\!\big[\|\nabla_q \ell_\eta(q,q)\|^2\big] + \mathbb{E}_{q\neq q'}\!\big[\max(0,c-\|\nabla_q \ell_\eta(q,q')\|^2)\big] $$
to enforce the same two conditions.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

# LAL parametrisation (η = {alpha_hat, c_hat})
def lal_loss(eta, pred, target):
    a = (2.5 - 0.01) * jax.nn.sigmoid(eta['alpha_hat']) + 0.01
    c = jax.nn.softplus(eta['c_hat']) + 1e-8
    d = pred - target
    return (jnp.abs(a - 2.0) / a) * (((d / c) ** 2 / jnp.abs(a - 2.0) + 1.0) ** (a / 2.0) - 1.0)

# inner PINN — `apply_fn(params, x)` returns the network output
def inner_loss(params, eta, apply_fn, x_f, x_b, u_b, lam):
    Lf = jnp.mean(lal_loss(eta, pde_residual(apply_fn, params, x_f, lam),
                           jnp.zeros_like(x_f[:, :1])))
    Lb = jnp.mean(lal_loss(eta, apply_fn(params, x_b), u_b))
    return Lf + Lb

def inner_unroll(params, eta, apply_fn, x_f, x_b, u_b, lam, J=10, eps2=1e-3):
    def body(p, _):
        g = jax.grad(inner_loss)(p, eta, apply_fn, x_f, x_b, u_b, lam)
        return jax.tree_util.tree_map(lambda pp, gg: pp - eps2 * gg, p, g), None
    final_p, _ = jax.lax.scan(body, params, None, length=J)
    return final_p

def outer_loss(eta, init_params, apply_fn, x_f, x_b, u_b, lam, x_val, u_val):
    p_J = inner_unroll(init_params, eta, apply_fn, x_f, x_b, u_b, lam)
    return jnp.mean((apply_fn(p_J, x_val) - u_val) ** 2)

eta       = {'alpha_hat': jnp.array(0.0), 'c_hat': jnp.array(0.0)}
meta_opt  = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
meta_state = meta_opt.init(eta)

@jax.jit
def meta_step(eta, meta_state, init_params, x_f, x_b, u_b, lam, x_val, u_val):
    g = jax.grad(outer_loss)(eta, init_params, model.apply,
                             x_f, x_b, u_b, lam, x_val, u_val)
    upd, meta_state = meta_opt.update(g, meta_state, eta)
    return optax.apply_updates(eta, upd), meta_state

for i in range(10_000):                          # I outer iters
    lam       = sample_task()                    # PDE parameter
    init_p    = model.init(jax.random.PRNGKey(i), jnp.zeros((1, d_in)))
    eta, meta_state = meta_step(eta, meta_state, init_p,
                                x_f, x_b, u_b, lam, x_val, u_val)
```

Recommended hyperparameters:
- Inner: `J = 10–50` SGD steps, Adam lr `ε_2 = 1e-3`.
- Outer: Adam lr `ε_1 = 1e-3`, 10 000 iters, grad clipping (to handle exploding meta-gradient).
- One task per outer step (`T=1`) is usually enough.
- LAL init: `α=2.01, c=1/√2` (≈MSE).
- For FFN losses add `L_{O,add}` with `c=10⁻²`.
- Meta-trained η can be re-used on different (wider/deeper) PINN architectures at test time.

## Results
On (a) discontinuous regression with varying frequency, (b) 1-D advection with varying IC, (c) steady reaction-diffusion with varying source, and (d) 1-D Burgers with varying viscosity, the meta-learned LAL and FFN losses give 2–10× smaller relative L2 than MSE/L1/Cauchy/Geman-McClure and beat the online OAL of Barron, including on OOD tasks and unseen architectures.
