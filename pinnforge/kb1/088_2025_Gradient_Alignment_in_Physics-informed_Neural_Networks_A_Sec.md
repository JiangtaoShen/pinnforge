---
slot: 88
title: "Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective"
authors: [Sifan Wang, Ananyae Kumar Bhartari, Bowen Li, Paris Perdikaris]
year: 2025
venue: "NeurIPS 2025 (arXiv:2502.00604)"
gitrepo: ""
---

## TL;DR
PINN training has two pathologies: magnitude imbalance between loss-term gradients (Type I, already addressed by adaptive weights) and **directional conflict** (Type II). The authors define a multi-vector alignment score, show first-order optimisers (Adam) keep it near zero, and prove that quasi-Newton **SOAP** (a Shampoo-Adam hybrid) is approximately Newton's preconditioner, giving near-perfect inter-step alignment - 2-10x accuracy gains across 10 PDE benchmarks including the first PINN turbulent flow at Re=10,000.

## Problem
Even after gradient *magnitude* balancing (Wang's lr-annealing) loss-term gradients still point in *different directions*, forcing Adam to zig-zag. Initialisation analysis on 1-D Laplace shows the intra-step alignment is a binary random variable - no first-order optimiser can fix this consistently.

## Method

### Gradient alignment score
For `n` loss-term gradients `g_1, ..., g_n` at step `k`:
$$ \mathcal{A}(g_1,\dots,g_n) = \frac{2\Big\|\sum_{i=1}^n g_i/\|g_i\|\Big\|^2}{n} - 1 \in [-1, 1] $$
(`= cos(g1, g2)` for `n=2`). Intra-step: `A(g_1^k, ..., g_n^k)`; inter-step: `A(g^{k-1}, g^k)`.

### SOAP as approximate Newton preconditioner
SOAP uses Shampoo's Kronecker-factored preconditioner `P` rotated by Adam in eigenbasis:
$$ \theta_{t+1} = \theta_t - \eta\, P^{-s}(\theta_t)\,\nabla L(\theta_t),\quad 0 \le s \le 1 $$
With `s=1`, `P ~ H` (Hessian), recovering Newton. Proposition: under L-smoothness, both intra- and inter-step alignment scores directly accelerate loss decay; SOAP yields the highest scores empirically.

### Practical pipeline (per PDE)
Combine SOAP with PirateNet backbone (RFF + adaptive residual init alpha=0 + Modified-MLP gating), exact periodic BC via Fourier features, NTK-style global loss balancing, and (optionally) causal training for time-dependent PDEs. Per-loss-term gradients are computed and weighted so `||g_i|| ~ const`.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from jax.flatten_util import ravel_pytree

def eigh_sqrt_inv(M, eps=1e-8):
    w, V = jnp.linalg.eigh(M)
    return V @ jnp.diag(1.0 / jnp.sqrt(w + eps)) @ V.T

def soap_init(params):
    """Per-parameter SOAP state. We treat each 2-D weight matrix separately."""
    def init_one(p):
        if p.ndim == 2:
            r, c = p.shape
            return dict(L=jnp.zeros((r, r)), R=jnp.zeros((c, c)),
                        Ql=jnp.eye(r), Qr=jnp.eye(c),
                        m=jnp.zeros_like(p), v=jnp.zeros_like(p), step=jnp.int32(0))
        return dict(m=jnp.zeros_like(p), v=jnp.zeros_like(p), step=jnp.int32(0))
    return jax.tree_util.tree_map(init_one, params)

def soap_update(grad, state, lr=1e-3, betas=(0.99, 0.999, 0.95), shampoo_freq=2):
    """Shampoo-Adam hybrid. Diagonalize preconditioner every shampoo_freq steps,
       then run Adam in that rotated eigenbasis."""
    if grad.ndim != 2:                              # vector/scalar: fallback to Adam
        m = betas[0]*state["m"] + (1-betas[0])*grad
        v = betas[1]*state["v"] + (1-betas[1])*grad*grad
        upd = m / (jnp.sqrt(v) + 1e-8)
        return -lr * upd, {**state, "m": m, "v": v, "step": state["step"]+1}
    L = betas[2]*state["L"] + (1-betas[2]) * (grad @ grad.T)
    R = betas[2]*state["R"] + (1-betas[2]) * (grad.T @ grad)
    new_step = state["step"] + 1
    Ql = jax.lax.cond(new_step % shampoo_freq == 0,
                      lambda _: eigh_sqrt_inv(L), lambda _: state["Ql"], None)
    Qr = jax.lax.cond(new_step % shampoo_freq == 0,
                      lambda _: eigh_sqrt_inv(R), lambda _: state["Qr"], None)
    g_rot = Ql @ grad @ Qr
    m = betas[0]*state["m"] + (1-betas[0])*g_rot
    v = betas[1]*state["v"] + (1-betas[1])*g_rot*g_rot
    upd_rot = m / (jnp.sqrt(v) + 1e-8)
    upd = Ql.T @ upd_rot @ Qr.T
    return -lr * upd, dict(L=L, R=R, Ql=Ql, Qr=Qr, m=m, v=v, step=new_step)

def alignment_score(grads_list):                    # grads_list: list of pytrees
    flats = [ravel_pytree(g)[0] for g in grads_list]
    units = [g / (jnp.linalg.norm(g) + 1e-12) for g in flats]
    s = sum(units); n = len(units)
    return 2 * (jnp.linalg.norm(s)**2) / n - 1

@jax.jit
def train_step(params, soap_state, batch):
    grads = jax.grad(total_loss)(params, batch)
    updates_and_states = jax.tree_util.tree_map(soap_update, grads, soap_state)
    updates = jax.tree_util.tree_map(lambda x: x[0], updates_and_states)
    soap_state = jax.tree_util.tree_map(lambda x: x[1], updates_and_states)
    return optax.apply_updates(params, updates), soap_state
```

Hyperparameters: PirateNet backbone (3 residual blocks = 9 layers; 4 for lid-driven cavity; width 256); RFF scale 2-10 per PDE; SOAP `lr=1e-3`, `beta1=0.99, beta2=0.999`, preconditioner update every 2 steps; Adam warm-up 5k steps then switch to SOAP; learning-rate annealing every 1k steps; causal training for time-dependent PDEs.

## Results
SOAP+PirateNet sets SOTA on 10 PDEs: Allen-Cahn 3.5e-6, KdV 3.4e-4, Wave, Burgers, Grey-Scott, Ginzburg-Landau, lid-driven cavity Re=5000, and - first PINN result on turbulent Navier-Stokes (Kolmogorov flow) at Re=10,000. Gains over Adam are 2-10x in relative L2. Empirically, SOAP's inter-step alignment is ~1 throughout training, vs ~0 for Adam.
