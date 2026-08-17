---
slot: 90
title: "Physics-informed neural networks with adaptive loss weighting algorithm for solving partial differential equations"
authors: [Bo Gao, Ruoxia Yao, Yan Li]
year: 2025
venue: Computers and Mathematics with Applications
gitrepo: ""
doi: 10.1016/j.camwa.2025.01.007
---

## TL;DR
Treat PINN training as 4-task multi-task learning (initial, lower-BC, upper-BC, PDE residual) and rescale per-task weights every N iterations from a running-average "learning speed" so the slowest task gets the largest weight and the fastest task gets weight 1.

## Problem
Vanilla PINNs minimize `L = L_u0 + L_lb + L_ub + L_f` with equal weights, but for nonlinear PDEs (Benjamin-Ono, Sine-Gordon, Mukherjee-Kundu) the residual loss `L_f` is orders of magnitude larger than the data losses, so the optimizer focuses on one term and ignores the others, producing 30-60% relative L2 error.

## Method
Define the per-task running-mean loss as a proxy for "training speed":
$$
\bar V_j = \frac{1}{N}\sum_{i=M-N+1}^{M} L_j^{(i)},\quad
\text{Ratio} = \frac{\max_j \bar V_j}{\min_j \bar V_j}
$$
Only update weights when speeds are very imbalanced (`Ratio > 10`). Compute a normalized relative score and set
$$
R_j = \frac{\bar V_j - \min_k \bar V_k}{\max_k \bar V_k - \min_k \bar V_k},\qquad
\lambda_j = 1 + \alpha R_j
$$
so the slowest task (largest `bar V_j`, `R_j=1`) gets `lambda = 1 + alpha`, the fastest (`R_j=0`) gets `lambda = 1`. The composite loss is
$$
\mathcal{L} = \lambda_1 L_{u_0} + \lambda_2 L_{lb} + \lambda_3 L_{ub} + \lambda_4 L_f
$$

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from collections import deque

class MLP(nn.Module):
    hidden: int = 50
    depth: int = 10
    out_dim: int = 1
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(self.out_dim)(x)

net = MLP()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

alpha   = 7.0                  # 4 for Sine-Gordon, 5 for MK, 7 for BO
N_avg   = 100                  # window for running mean
lam     = jnp.ones(4)
history = [deque(maxlen=N_avg) for _ in range(4)]

def task_losses(params, X0, Xlb, Xub, Xf):
    L_u0 = jnp.mean((net.apply(params, X0) - u0_target)**2)
    L_lb = jnp.mean((net.apply(params, Xlb) - g_lb)**2)
    L_ub = jnp.mean((net.apply(params, Xub) - g_ub)**2)
    L_f  = pde_residual_loss(params, Xf)        # autograd via jax.grad
    return jnp.stack([L_u0, L_lb, L_ub, L_f])

@jax.jit
def train_step(params, opt_state, lam, X0, Xlb, Xub, Xf):
    def total(p):
        Ls = task_losses(p, X0, Xlb, Xub, Xf)
        return jnp.sum(lam * Ls), Ls
    (L, Ls), grads = jax.value_and_grad(total, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, Ls

for it in range(max_iter):
    params, opt_state, Ls = train_step(params, opt_state, lam, X0, Xlb, Xub, Xf)
    for j in range(4): history[j].append(float(Ls[j]))

    if it % N_avg == 0 and it > 0:
        V = jnp.array([sum(h)/len(h) for h in history])
        ratio = V.max() / jnp.clip(V.min(), 1e-12)
        if ratio > 10.0:
            R = (V - V.min()) / (V.max() - V.min() + 1e-12)
            lam = 1.0 + alpha * R          # fastest -> 1, slowest -> 1+alpha
```

Hyperparameters: tanh MLP, 5-10 hidden layers of 50-100 units, Adam, `N_0=50-2000`, `N_b=50-2000`, `N_f=10k-20k`, `alpha in {4,5,7}` per PDE, weight refresh window `N=100`, Ratio threshold 10.

## Results
Compared with vanilla PINN on three nonlinear wave PDEs (same network/sampling): Benjamin-Ono solitary wave L2 6.44e-1 -> 1.49e-2; Sine-Gordon breather 2.13e-1 -> 2.07e-2; Mukherjee-Kundu breather 4.11e-1 -> 1.00e-1. Loss curves confirm the four task losses become same-order-of-magnitude under APINN.
