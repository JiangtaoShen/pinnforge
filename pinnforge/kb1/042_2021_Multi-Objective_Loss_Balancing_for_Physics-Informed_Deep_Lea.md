---
slot: 042
title: "Multi-Objective Loss Balancing for Physics-Informed Deep Learning"
authors: [Rafael Bischof, Michael A. Kraus]
year: 2021
venue: Computer Methods in Applied Mechanics and Engineering
gitrepo: "https://github.com/rbischof/relative_balancing"
---

## TL;DR
The paper benchmarks three existing PINN loss-balancing schemes (Learning-Rate Annealing, GradNorm, SoftAdapt) and proposes ReLoBRaLo (Relative Loss Balancing with Random Lookback), a cheap loss-only scheme that combines softmax normalisation, relative progress versus the previous step, exponential moving average, and a Bernoulli "saudade" lookback to the initial losses. ReLoBRaLo matches or beats the alternatives on Burgers, Kirchhoff and Helmholtz while costing essentially nothing extra.

## Problem
PINN composite losses (PDE residual + BC + IC + data) have terms with very different units, so their gradient magnitudes differ by orders of magnitude. Naive equal weighting causes one term to dominate. Existing fixes either need a separate backward pass per term (GradNorm), are unbounded (LRAnnealing), or ignore long-term progress (SoftAdapt).

## Method
Maintain time-varying weights `λ_i(t)` for each of `k` losses in
$$ L(\theta) = \sum_{i=1}^{k} \lambda_i(t)\,\mathcal{L}_i(\theta) $$
ReLoBRaLo updates `λ_i(t)` purely from loss values (no extra grad evaluations) and combines three ideas:

1. **Softmax over relative progress** vs reference step `t'` (acts on loss ratios, bounded):
$$ \lambda_i^{\text{bal}}(t,t') = k\cdot\frac{\exp\!\big(\mathcal{L}_i(t)\,/\,(T\,\mathcal{L}_i(t'))\big)}{\sum_j \exp\!\big(\mathcal{L}_j(t)\,/\,(T\,\mathcal{L}_j(t'))\big)} $$
2. **Random lookback** (saudade) — with probability `ρ` use the previous step, with `1-ρ` look back to initialisation `t'=0`:
$$ \lambda_i^{\text{hist}}(t) = \rho\,\lambda_i(t-1) + (1-\rho)\,\lambda_i^{\text{bal}}(t,0) $$
3. **EMA blend** with new measurement:
$$ \lambda_i(t) = \alpha\,\lambda_i^{\text{hist}}(t) + (1-\alpha)\,\lambda_i^{\text{bal}}(t,t-1) $$

Recommended hyperparameters: `α = 0.999`, temperature `T = 0.1`, `ρ` Bernoulli with mean `0.9999` (i.e. usually use the last-step balance, occasionally re-anchor to initialisation).

```python
import jax, jax.numpy as jnp
import optax
from functools import partial

def softmax_balance(L_now, L_ref, k, T):
    z = L_now / (T * L_ref + 1e-12)
    return k * jax.nn.softmax(z)

def relobralo_update(state, L_now, key, alpha=0.999, T=0.1, rho_mean=0.9999):
    lam, L_prev, L_init = state['lam'], state['L_prev'], state['L_init']
    k = lam.shape[0]
    rho      = jax.random.bernoulli(key, p=rho_mean).astype(L_now.dtype)
    lam_step = softmax_balance(L_now, L_prev, k, T)
    lam_hist = rho * lam + (1.0 - rho) * softmax_balance(L_now, L_init, k, T)
    lam_new  = alpha * lam_hist + (1.0 - alpha) * lam_step
    return {'lam': lam_new, 'L_prev': L_now, 'L_init': L_init}

def init_relobralo(k):
    return {'lam':    jnp.ones(k),
            'L_prev': jnp.ones(k),
            'L_init': jnp.ones(k)}

def losses_vec(params, x_f, x_b, u_b, x_ic, u_ic):
    return jnp.array([pde_residual_loss(params, x_f),
                      bc_loss(params, x_b, u_b),
                      ic_loss(params, x_ic, u_ic)])

def weighted_total(params, lam, x_f, x_b, u_b, x_ic, u_ic):
    return jnp.sum(lam * losses_vec(params, x_f, x_b, u_b, x_ic, u_ic))

opt = optax.adam(1e-3)
opt_state    = opt.init(params)
balance_st   = init_relobralo(3)                          # PDE, BC, IC

@jax.jit
def step(params, opt_state, balance_st, key, x_f, x_b, u_b, x_ic, u_ic):
    L_now      = jax.lax.stop_gradient(losses_vec(params, x_f, x_b, u_b, x_ic, u_ic))
    # warm-start L_init on the first call (assumed pre-seeded after step 0)
    balance_st = relobralo_update(balance_st, L_now, key)
    grads      = jax.grad(weighted_total)(params, balance_st['lam'],
                                          x_f, x_b, u_b, x_ic, u_ic)
    upd, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, balance_st

key = jax.random.PRNGKey(0)
for it in range(50_000):
    key, sub = jax.random.split(key)
    params, opt_state, balance_st = step(params, opt_state, balance_st, sub,
                                         x_f, x_b, u_b, x_ic, u_ic)
```

Comparison summary (the three baselines also re-implemented):
- LRAnnealing: `λ_i = max|∇L_PDE| / mean|∇L_i|` + EMA. Unbounded; one extra backward pass per term.
- GradNorm: trainable `λ_i` updated by a secondary optimiser to equalise normalised gradient rates.
- SoftAdapt: `λ_i ∝ exp(T·(L_i(t) − L_i(t-1)))`, no gradient stats.
- ReLoBRaLo: combines all three insights, no extra backward passes.

## Results
On forward Burgers, Helmholtz, Kirchhoff plate bending, and inverse Burgers/Kirchhoff, ReLoBRaLo gives the lowest test L2 error in most cases and incurs <1% training-time overhead, versus 1.5–3× for GradNorm. LRAnnealing is competitive but unstable; SoftAdapt is fast but worse than ReLoBRaLo. Equal weights consistently fail.
