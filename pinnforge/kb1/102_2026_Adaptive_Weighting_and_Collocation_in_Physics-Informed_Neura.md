---
slot: 102
title: "Adaptive Weighting and Collocation in Physics-Informed Neural Networks for Chemical Process Modeling"
authors: [Guoquan Wu, Keerthana Vellayappan, Yao Shi, Zhe Wu]
year: 2026
venue: Industrial & Engineering Chemistry Research 65 (2026) 7092-7107
gitrepo: ""
doi: 10.1021/acs.iecr.6c00048
---

## TL;DR
PaRS-PINN splits the domain into M K-means regions, treats each region's residual `L_r^(m)` plus `L_IC`, `L_BC`, `L_data` as separate objectives in a multi-objective problem, runs NSGA-II to sample a small Pareto front, and from front statistics computes saliency-based softmax weights (Pareto-Guided Weighting). The same weights drive a region-wise resampling probability so collocation density follows learning difficulty.

## Problem
For stiff chemical engineering ODE/PDE systems (CSTR with Arrhenius kinetics, Cahn-Hilliard phase separation, plug flow advection), uniform-sampled PINNs with fixed weights underperform: errors concentrate near phase fronts or steep gradients; residual-only adaptive schemes (RAR, R3) greedily sample sharp regions and forget the stable bulk; loss-weight imbalance among inner residual sub-objectives is ignored.

## Method
Partition collocation points into M nonoverlapping subdomains `Omega_m` (default K-means on normalized coords, M=3) with shared weight `alpha_m`:
$$
\mathcal L_r^{(m)} = \frac{1}{N_m}\sum_{(x,t)\in\Omega_m}\|\mathcal R[u_\theta;x,t,\lambda]\|^2,\quad
\mathcal L = \sum_{m=1}^M \alpha_m \mathcal L_r^{(m)} + \alpha_b \mathcal L_{BC} + \alpha_i \mathcal L_{IC} + \alpha_d \mathcal L_{data}
$$

Every `N_inner` epochs:

**1. NSGA-II Pareto search.** Initialize population `P_0` from current `(theta, lambda)`, evolve with crossover/mutation for `N_g` generations on the `(M+3)`-objective vector `F = (L_r^(1), ..., L_r^(M), L_IC, L_BC, L_data)`. Pop size `N_p=50`, `N_g=50` suffice (sensitivity analysis).

**2. Pareto-Guided Weights.** Normalize the `N_p x N_o` Pareto matrix per objective:
$$
F^{\text{norm}}_{ij} = \frac{F_{ij}-\min_i F_{ij}}{\max_i F_{ij}-\min_i F_{ij}}
$$
Variability `sigma_j = std_i(F_ij_norm)` and improvement ratio `r_j = mean_i(F_ij) / F_j^pre` where `F_j^pre` is the historical minimum. Saliency
$$
s_j = \sigma_j / r_j,\qquad
\alpha_j = \frac{\exp(s_j/\tau_j)}{\sum_k \exp(s_k/\tau_k)}
$$
with temperature `tau_data = 0.5` while `L_data > epsilon` (e.g. 1e-4) and `tau_j = 1` after.

**3. Weight-informed resampling.** Convert weights to region probabilities and draw `N_m = N_r rho_m` collocation points per region:
$$
\rho_m = \frac{\alpha_m^\gamma}{\sum_k \alpha_k^\gamma}\quad(\gamma=1\text{ default})
$$
These weights / points are then fixed for the next `N_inner` Adam steps that update `(theta, lambda)`.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from sklearn.cluster import KMeans
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as moo_minimize
from pymoo.core.problem import Problem

def regions(X_pool, M=3):
    km = KMeans(n_clusters=M, n_init=10).fit(np.asarray(X_pool))
    return [np.where(km.labels_ == m)[0] for m in range(M)]

def losses_from_params(params, lam, X_pool, idx_list, X_ic, X_bc, X_data, u_data):
    losses = []
    for idx in idx_list:
        losses.append(pde_residual_mse(params, X_pool[idx], lam))
    losses += [ic_mse(params, X_ic), bc_mse(params, X_bc), data_mse(params, X_data, u_data)]
    return jnp.stack(losses)

def pgw_weights(F_mat, F_pre, temps):
    Fn = (F_mat - F_mat.min(0)) / (F_mat.max(0) - F_mat.min(0) + 1e-12)
    sigma = Fn.std(0)
    r = F_mat.mean(0) / (F_pre + 1e-12)
    s = sigma / (r + 1e-12)
    w = np.exp(s/temps); return w / w.sum()

# Outer loop
M, N_inner = 3, 1000
idx_list = regions(X_pool, M)
alphas   = np.ones(M + 3) / (M + 3)
F_pre    = np.full(M + 3, np.inf)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, lam, X_pool, idx_list, X_ic, X_bc, X_data, u_data, alphas):
    def total(p):
        L_parts = losses_from_params(p, lam, X_pool, idx_list,
                                      X_ic, X_bc, X_data, u_data)
        return jnp.sum(alphas * L_parts)
    g = jax.grad(total)(params)
    updates, opt_state = optimizer.update(g, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

for epoch in range(max_iter):
    if epoch % N_inner == 0 and epoch > 0:
        # 1) NSGA-II population centered on (theta, lam)
        # ... evaluate F at small perturbations of (theta, lam), get F_mat (N_p, M+3)
        F_pre = np.minimum(F_pre, F_mat.min(0))
        temps = np.where(np.arange(M + 3) == M + 2,
                         0.5 if float(L_data) > 1e-4 else 1.0, 1.0)
        alphas = pgw_weights(F_mat, F_pre, temps)
        rho = alphas[:M] / alphas[:M].sum()
        new = []
        for m in range(M):
            Nm = max(1, int(round(N_r * rho[m])))
            new.append(resample_in_region(m, Nm))
        X_pool = jnp.concatenate(new, 0); idx_list = regions(X_pool, M)

    params, opt_state = step(params, opt_state, lam, X_pool, idx_list,
                              X_ic, X_bc, X_data, u_data, jnp.asarray(alphas))
```

Hyperparameters: M=3 K-means regions, NSGA-II pop 50 / generations 50, `gamma=1`, `N_inner=1000`, Adam for inner loop, default tanh MLP for `u_theta`. Inverse problems jointly train PDE parameters `lambda` and weights `theta`.

## Results
Three case studies: (1) nonisothermal CSTR with second-order Arrhenius kinetics (ODE, recurrent PINN); (2) Cahn-Hilliard phase separation in multicomponent mixture (PDE); (3) plug-flow reactor advection. PaRS-PINN converges faster than uniform-sampling baseline and the R3 algorithm, with smaller `(theta, lambda)` recovery errors and better region-balanced residuals; the inverse-problem improvements are largest. NSGA-II hyperparams (`N_p, N_g`) insensitive beyond ~50 each; `gamma=1` is the recommended default.
