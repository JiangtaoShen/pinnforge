---
slot: 78
title: "Multi-level physics-informed deep learning for solving partial differential equations in computational structural mechanics"
authors: [Weiwei He, Jinzhao Li, Xuan Kong, Lu Deng]
year: 2024
venue: "Communications Engineering 3 (2024)"
gitrepo: "https://github.com/he-weiwei/ml-PINN"
doi: "10.1038/s44172-024-00303-3"
---

## TL;DR
For bending of beams and shells (governed by 4th-order nonlinear PDEs), ml-PINN replaces a single high-order network by an *aggregation* of several small networks, each predicting one intermediate mechanical quantity (displacement, rotation, curvature, moment, shear). Loss terms now contain only first- or second-order derivatives, eliminating the magnitude mismatch that breaks single-net PINNs.

## Problem
The Euler-Bernoulli equation `EI d^4 w/dx^4 = q` and the Kirchhoff plate equation `D nabla^4 w = q` need 4th-order autodiff on a single net. Loss components per derivative order differ by many orders of magnitude, leading to non- or local-convergence and explosive runtime.

## Method
Decompose the 4th-order PDE into a chain of first/second-order relations and assign each variable its own network. Beam (1-D):
$$ q = \frac{dF_Q}{dx},\quad F_Q = \frac{dM}{dx},\quad M = EI\kappa,\quad \kappa = \frac{d\theta}{dx},\quad \theta = \frac{dw}{dx} $$
giving 5 networks `w_NN, theta_NN, kappa_NN, M_NN, F_Q_NN`. Plate (2-D) is similar with `q = nabla^2 M`, `M = D kappa(w)`, `kappa = nabla^2 w`. The combined loss is a *sum of MSEs of first/second-order residuals* plus boundary MSEs imposed directly on each sub-network's output (no derivatives on BCs):
$$ \mathcal{L} = \sum_{\text{eqns}} \|\partial v_i - v_{i+1}\|^2 + \sum_{\text{BCs}} \|v_i - \bar v_i\|^2 $$

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    H: int = 20
    depth: int = 3
    out_dim: int = 1
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.H)(x))
        return nn.Dense(self.out_dim)(x)

class MLPINNBeam(nn.Module):
    EI: float = 1.0
    @nn.compact
    def __call__(self, x):                          # returns dict of 5 outputs
        return {
            "w" : MLP(name="w_NN")(x),
            "t" : MLP(name="t_NN")(x),
            "k" : MLP(name="k_NN")(x),
            "M" : MLP(name="M_NN")(x),
            "Fq": MLP(name="Fq_NN")(x),
        }

def d(out_fn, x):                                   # scalar-output, scalar input
    return jax.vmap(jax.grad(lambda xi: out_fn(xi).squeeze()))(x)

def loss(params, x, q, x_bc, bc, EI=1.0):
    apply = lambda xi: MLPINNBeam(EI=EI).apply(params, xi)
    w_of  = lambda xi: apply(xi)["w"].squeeze()
    t_of  = lambda xi: apply(xi)["t"].squeeze()
    k_of  = lambda xi: apply(xi)["k"].squeeze()
    M_of  = lambda xi: apply(xi)["M"].squeeze()
    Fq_of = lambda xi: apply(xi)["Fq"].squeeze()
    dw  = jax.vmap(jax.grad(w_of))(x)
    dt  = jax.vmap(jax.grad(t_of))(x)
    dM  = jax.vmap(jax.grad(M_of))(x)
    dFq = jax.vmap(jax.grad(Fq_of))(x)
    t   = jax.vmap(t_of)(x);  k = jax.vmap(k_of)(x)
    M   = jax.vmap(M_of)(x);  Fq= jax.vmap(Fq_of)(x)
    L_pde = (jnp.mean((dw - t)**2) + jnp.mean((dt - k)**2)
           + jnp.mean((EI*k - M)**2) + jnp.mean((dM - Fq)**2)
           + jnp.mean((dFq - q)**2))
    # BCs imposed directly on each network's output
    L_bc = 0.0
    for v, target in bc.items():
        L_bc = L_bc + jnp.mean((apply(x_bc[v])[v].squeeze() - target)**2)
    return L_pde + L_bc

optimizer = optax.adam(1e-3); opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, x, q, x_bc, bc):
    grads = jax.grad(loss)(params, x, q, x_bc, bc)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state

for it in range(20000):
    params, opt_state = step(params, opt_state, x_col, q_col, x_bc, bc_vals)
```

Hyperparameters: tanh MLPs 3 x 10 per sub-network; 50-100 collocation points (beam) / 1000 interior + 50/edge (shell); Adam lr `1e-3`; ~`2e4`-`5e4` steps. Plate uses 3 networks `w_NN, kappa_NN(x,y), M_NN(x,y)`.

## Results
Beam (4 loading cases) relative error <0.5%; plate (4 cases) <2% (largest at clamped corners). Compared with single-net PINN: ~4x faster convergence and notably more accurate. Once trained with `(u, q)` as inputs, ml-PINN generalises to new boundary conditions and loads without retraining.
