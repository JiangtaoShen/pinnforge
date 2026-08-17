---
slot: 9
title: "Transfer learning enhanced physics informed neural network for phase-field modeling of fracture"
authors: [Somdatta Goswami, Souvik Chakraborty, Cosmin Anitescu, Timon Rabczuk]
year: 2019
venue: "Theoretical and Applied Fracture Mechanics (arXiv:1907.02531)"
gitrepo: ""
---

## TL;DR
A Deep-Energy-Method PINN for phase-field brittle fracture: minimise the *variational energy* (elastic + fracture surface energy) rather than the residual; enforce Dirichlet BCs *exactly* via output reshaping; integrate with NURBS-element Gauss-Legendre quadrature with adaptive (quad/oct-tree) refinement near the crack; accelerate the multi-step load incrementation by *transfer learning* (retrain only the last layer between consecutive load steps).

## Problem
Phase-field fracture needs to be solved at hundreds of load increments; each is itself a coupled PDE for displacement `u` and damage `phi`. Pure residual-PINN is slow (higher derivatives) and requires Lagrange/penalty balance. Re-training from scratch at every load step is prohibitive.

## Method
Networks: one MLP `u_NN(x)` -> displacement, one MLP `phi_NN(x)` -> phase field. Each output is post-processed so Dirichlet BCs hold exactly, e.g. `u(x) = u_bar(x) + D(x) * u_NN(x)` with `D = 0` on `dOmega_D`.

Loss = total potential energy (no boundary penalty, no PDE residual):
$$
\mathcal{L}(\theta) = \int_\Omega \bigl[g(\phi)\Psi_0^+(\varepsilon) + \Psi_0^-(\varepsilon)\bigr]\,d\Omega + \int_\Omega \bigl[\tfrac{G_c}{2 l_0}\phi^2 + \tfrac{G_c l_0}{2}|\nabla\phi|^2 + g(\phi)\mathcal{H}\bigr]\,d\Omega
$$
where `g(phi) = (1 - phi)^2`, `Psi_0^+/-` are tension/compression splits of strain energy, `H(x,t) = max_{s<=t} Psi_0^+` (irreversibility / strain-history), `G_c` is critical energy release rate, `l_0` length scale.

Quadrature: NURBS-described geometry -> partition into elements -> Gauss-Legendre points per element, refined near the (evolving) crack via quad/oct-tree subdivision. So integrals become
$$
\int_\Omega f\,d\Omega \approx \sum_e \sum_{q} f(x_{eq})\,w_{eq}\,|J(x_{eq})|
$$

Transfer learning across load steps: at load step `k+1`, freeze all parameters of `u_NN` and `phi_NN` except the *output layer*; warm-start from converged state at step `k`; only the last linear layer trains, typically a few hundred Adam steps. Allows much larger load increments and ~5-10x wall-clock speed-up over full re-training.

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    h: int = 50
    depth: int = 4
    out_dim: int = 2
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.h)(x))
        return nn.Dense(self.out_dim)(x)

u_net   = MLP(out_dim=2)
phi_net = MLP(out_dim=1)
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
params = {"u":   u_net.init(key1, jnp.zeros((1, 2))),
          "phi": phi_net.init(key2, jnp.zeros((1, 2)))}

def disp(params, xy):                          # hard Dirichlet
    return u_bar(xy) + D(xy) * u_net.apply(params["u"], xy)

def energy_loss(params, xy_gp, w_gp, H_field):
    u = disp(params, xy_gp)
    phi = jax.nn.sigmoid(phi_net.apply(params["phi"], xy_gp))
    grad_u = jax.vmap(lambda x: jax.jacrev(lambda y: disp(params, y[None])[0])(x))(xy_gp)
    eps = sym_grad(grad_u)
    Pp, Pm = strain_energy_split(eps)
    g = (1 - phi) ** 2
    grad_phi = jax.vmap(lambda x: jax.grad(
        lambda y: jax.nn.sigmoid(phi_net.apply(params["phi"], y[None]))[0, 0])(x))(xy_gp)
    fe = (g[:, 0] * Pp + Pm) * w_gp
    fc = (Gc / (2 * l0) * phi[:, 0] ** 2 + (Gc * l0 / 2) * (grad_phi ** 2).sum(-1)
          + g[:, 0] * H_field) * w_gp
    return jnp.sum(fe + fc)

# Load-step loop with transfer learning: build a per-step partition selector
# that masks gradients to all-but-last-layer when k > 0.
def make_mask(params, k):
    if k == 0:
        return jax.tree_util.tree_map(lambda _: True, params)
    mask = jax.tree_util.tree_map(lambda _: False, params)
    # Mark only the final Dense layer of each subnet as trainable.
    mask["u"]["params"][f"Dense_{u_net.depth}"] = jax.tree_util.tree_map(lambda _: True,
                                                                        params["u"]["params"][f"Dense_{u_net.depth}"])
    mask["phi"]["params"][f"Dense_{phi_net.depth}"] = jax.tree_util.tree_map(lambda _: True,
                                                                            params["phi"]["params"][f"Dense_{phi_net.depth}"])
    return mask

H = init_history()
for k, load in enumerate(load_steps):
    mask = make_mask(params, k)
    optimizer = optax.masked(optax.adam(1e-3 if k == 0 else 1e-4), mask)
    opt_state = optimizer.init(params)
    n_iter = 2000 if k == 0 else 300

    @jax.jit
    def step(p, s, H_):
        g = jax.grad(energy_loss)(p, xy_gp, w_gp, H_)
        u, s = optimizer.update(g, s, p)
        return optax.apply_updates(p, u), s

    for _ in range(n_iter):
        params, opt_state = step(params, opt_state, H)
    H = jnp.maximum(H, Psi_plus_at(params, xy_gp))
```

Recommended: depth 4-5, width 30-50, tanh; Adam `lr=1e-3` for step 0 (a few thousand iters), `lr=1e-4` and ~300 iters for subsequent steps. `l_0 ~ 2 * h_quad`. Gauss order 3-4 per direction.

## Results
On single-edge-notched tension/shear and L-shape panel, the energy-PINN with transfer learning matches reference phase-field FEM crack paths within ~3% in load-displacement curves, while running ~5-10x faster than retraining from scratch and reaching lower energy than residual-PINN on the first two benchmarks.
