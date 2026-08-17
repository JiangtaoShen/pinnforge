---
slot: 50
title: "Understanding and Mitigating Gradient Pathologies in Physics-Informed Neural Networks"
authors: [Sifan Wang, Yujun Teng, Paris Perdikaris]
year: 2021
venue: SIAM J. Sci. Comput. (arXiv:2001.04536)
gitrepo: "https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs"
doi: 10.1137/20M1318043
---

## TL;DR
Vanilla PINNs fail on stiff problems because the back-propagated gradients of the PDE-residual loss are 1–3 orders of magnitude smaller than those of the boundary/initial losses. Two fixes are proposed and composed: an **adaptive learning-rate annealing** that rebalances loss weights from live gradient statistics, and an **improved fully-connected architecture** with multiplicative input embeddings and gated residual hidden updates.

## Problem
For composite PINN losses `L = L_r + Σ λ_i L_i`, gradients `∇θ L_r` and `∇θ L_i` typically differ by 10²–10³ in magnitude. SGD then "sees" only the dominant term and the under-weighted loss never converges (e.g. vanilla PINN on 2-D Helmholtz with a_1=1, a_2=4 stalls at ~1.1e-1 relative L2). The authors trace this to numerical stiffness of the gradient-flow ODE governing PINN training — the largest Hessian eigenvalue grows during training.

## Method

### A. Learning-rate annealing for loss weights

Maintain one weight `λ_i` per non-residual loss term. Every step, sample gradient statistics and EMA-update the weights.

Update rule (Algorithm 1):
$$
\hat{\lambda}_i = \frac{\max_\theta |\nabla_\theta \mathcal{L}_r(\theta_n)|}{\overline{|\nabla_\theta \mathcal{L}_i(\theta_n)|}}, \qquad
\lambda_i \leftarrow (1-\alpha)\lambda_i + \alpha\hat{\lambda}_i
$$

Recommended: `α = 0.9`, learning rate `η = 1e-3`.

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

def grad_max(loss_fn, params, *args):
    g = jax.grad(loss_fn)(params, *args)
    flat = jnp.concatenate([jnp.abs(x).ravel() for x in jax.tree_util.tree_leaves(g)])
    return jnp.max(flat)

def grad_mean(loss_fn, params, *args):
    g = jax.grad(loss_fn)(params, *args)
    flat = jnp.concatenate([jnp.abs(x).ravel() for x in jax.tree_util.tree_leaves(g)])
    return jnp.mean(flat)

alpha = 0.9
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)
lambdas = {"bc": jnp.array(1.0), "ic": jnp.array(1.0)}

@jax.jit
def update_lambdas(params, lambdas, x_r, x_bc, u_bc, x_ic, u_ic):
    g_r  = grad_max (loss_residual, params, x_r)
    g_bc = grad_mean(loss_bc,      params, x_bc, u_bc)
    g_ic = grad_mean(loss_ic,      params, x_ic, u_ic)
    lambdas = {
        "bc": (1-alpha)*lambdas["bc"] + alpha*(g_r / (g_bc + 1e-12)),
        "ic": (1-alpha)*lambdas["ic"] + alpha*(g_r / (g_ic + 1e-12)),
    }
    return lambdas

@jax.jit
def train_step(params, opt_state, lambdas, x_r, x_bc, u_bc, x_ic, u_ic):
    def total(p):
        return (loss_residual(p, x_r)
                + lambdas["bc"] * loss_bc(p, x_bc, u_bc)
                + lambdas["ic"] * loss_ic(p, x_ic, u_ic))
    grads = jax.grad(total)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

### B. Improved fully-connected architecture

Two "transformer" projections `U, V` of the inputs are gated into every hidden layer by element-wise multiplication (gated residual block, no attention).

$$
U = \phi(W_u X + b_u), \quad V = \phi(W_v X + b_v)
$$
$$
H^{(1)} = \phi(W^{(1)} X + b^{(1)})
$$
$$
Z^{(l)} = \phi(W^{(l)} H^{(l)} + b^{(l)}), \quad H^{(l+1)} = (1 - Z^{(l)}) \odot U + Z^{(l)} \odot V
$$

Output `f(X) = W H^{(L)} + b`. ~10% parameter overhead vs plain MLP; reduces the Hessian's max eigenvalue.

```python
class ImprovedMLP(nn.Module):
    hidden: int
    out_dim: int
    depth: int
    act: callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        U = self.act(nn.Dense(self.hidden, name="Wu")(x))
        V = self.act(nn.Dense(self.hidden, name="Wv")(x))
        H = self.act(nn.Dense(self.hidden, name="W0")(x))
        for i in range(self.depth - 1):
            Z = self.act(nn.Dense(self.hidden, name=f"W{i+1}")(H))
            H = (1.0 - Z) * U + Z * V
        return nn.Dense(self.out_dim, name="Wout")(H)
```

The two methods compose: use `ImprovedMLP` as backbone, train with annealed λ.

## Results
On 2-D Helmholtz (a_1=1, a_2=4), annealing alone improves relative-L2 from ~1.1e-1 to ~1.3e-2; adding the improved architecture pushes it to 3.7e-3. Similar 50–100× gains on Klein-Gordon and lid-driven cavity Navier-Stokes (Re=100), with <5% training overhead from the extra autograd calls.
