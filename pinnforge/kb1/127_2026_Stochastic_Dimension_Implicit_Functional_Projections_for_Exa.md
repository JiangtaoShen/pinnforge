---
slot: 127
title: "Stochastic Dimension Implicit Functional Projections for Exact Integral Conservation in High-Dimensional PINNs"
authors: [Zhangyong Liang]
year: 2026
venue: arXiv:2603.29237
gitrepo: ""
---

## TL;DR
SDIFP enforces *exact* mass and energy integral conservation in PINN solutions of high-dimensional PDEs by applying a global affine transformation `ũ(x,t) = α(t) u_raw(x,t;θ) + β(t)` to the network output, with `α, β` solved in closed-form from Monte Carlo estimates of the first/second spatial moments. Combined with a double-stochastic unbiased gradient estimator (DS-UGE), this collapses memory complexity from `O(M·d)` to `O(N·|I|)` while remaining mesh-free and preserving O(1) inference.

## Problem
Soft-penalty conservation in PINNs permits violations and yields artificial dissipation. Existing hard-projection PINN-proj forces a discrete Riemann sum, requiring a uniform grid that fails the mesh-free promise and explodes in high dimensions. Implicit-optimization layers (Πnet via Douglas-Rachford + IFT) only work for *convex* constraints — but energy `∫ u² dx = c_2(t)` is a non-convex hyper-ellipsoid. Reverse-mode AD through high-order operators in d dimensions also costs `O(M·N_L)` memory.

## Method

### A. Affine functional projection
Define
$$\tilde u(x,t) = \alpha(t)\,u_{\text{raw}}(x,t;\theta) + \beta(t)$$
Imposing both linear and quadratic integral constraints gives a 2x2 algebraic system. With `c̄_1, c̄_2` the domain-averaged target moments and `μ_1, μ_2` the empirical first/second moments of `u_raw`:
$$\alpha^* = \sqrt{\frac{\bar c_2 - \bar c_1^2}{\mu_2 - \mu_1^2}},\qquad \beta^* = \bar c_1 - \alpha^* \mu_1$$
Phase choice `α>0` removes sign ambiguity. Cross-terms cancel identically — the hyper-ellipsoid non-convexity is bypassed by scaling rather than projecting.

### B. Detached Monte Carlo quadrature
Evaluate `μ_1, μ_2` over a *large* `M ≈ 10⁵` Sobol point set `S_MC`, detached from the AD graph. Mesh-free and dimension-agnostic. Per-batch projection uses the *same batch* to compute `δ = c̄_1 − ū({x_i})`, guaranteeing exact algebraic conservation on each minibatch (residual `ε=0`).

### C. Double-stochastic unbiased gradient estimator (DS-UGE)
For the implicit gradients `∇_θ α*, ∇_θ β*`, exploit the closed form: the Jacobians `∂α*/∂μ_1, ∂α*/∂μ_2` etc. are analytic. The moment gradients are estimated on the SAME small training mini-batch `S_batch` (size N≈10³):
$$\nabla_\theta \mu_1 = \frac1N\!\!\sum_{x_j\in S_{\text{batch}}}\!\!\nabla_\theta u_{\text{raw}}(x_j),\quad \nabla_\theta \mu_2 = \frac2N\!\!\sum_{x_j\in S_{\text{batch}}}\!\!u_{\text{raw}}(x_j)\nabla_\theta u_{\text{raw}}(x_j)$$
For high-d linear operators, additionally subsample `|I|` differential dimensions (SDGD-style) → composite gradient `g_{I,J}(θ)` is unbiased by Fubini.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class SDIFP(nn.Module):
    hidden: int = 128
    depth: int = 4
    @nn.compact
    def __call__(self, xt):
        h = xt
        for _ in range(self.depth):
            h = jnp.tanh(nn.Dense(self.hidden)(h))
        return nn.Dense(1)(h).squeeze(-1)

def raw_apply(params, xt):
    return net.apply(params, xt)

def project(params, xt_batch, xt_mc, c1_bar, c2_bar, eps=1e-8):
    u = raw_apply(params, xt_batch)                            # gradient-bearing
    u_mc = jax.lax.stop_gradient(raw_apply(params, xt_mc))
    mu1 = u_mc.mean()
    mu2 = (u_mc ** 2).mean()
    var = jnp.clip(mu2 - mu1 ** 2, eps, None)
    alpha = jnp.sqrt((c2_bar - c1_bar ** 2) / var)
    beta  = c1_bar - alpha * mu1
    return alpha * u + beta, alpha, beta, mu1, mu2

def sdifp_loss(params, xt_batch, xt_mc, c1_bar, c2_bar, pde_residual_op):
    def res_fn(p):
        u_tilde, alpha, beta, _, _ = project(p, xt_batch, xt_mc, c1_bar, c2_bar)
        return pde_residual_op(u_tilde, xt_batch)              # uses alpha, beta inside
    res = res_fn(params)
    return jnp.mean(res ** 2)

net = SDIFP()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
opt = optax.adam(optax.linear_schedule(1e-3, 0.0, 10_000))
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, xt_batch, xt_mc, c1_bar, c2_bar):
    loss, grads = jax.value_and_grad(sdifp_loss)(
        params, xt_batch, xt_mc, c1_bar, c2_bar, pde_residual_op)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss
```

### D. Compatibility
Native for periodic / homogeneous Neumann BCs. For Dirichlet, multiply β by a distance-function mask `φ(x)|_∂Ω = 0`.

Hyperparameters: 4-layer MLP x 128 tanh; Adam lr=1e-3 → 0 linear over 10k epochs; `M=1e5` Sobol MC points, residual batch `N=100`, dimension subset `|I|=100`. ε=1e-8 numerical floor.

## Results
On 1D Advection / Reaction-Diffusion / Wave / KdV with both fixed-grid and random collocation: SDIFP keeps |C_pred(t) − C_true(t)| at 1e-6-1e-7 throughout time evolution; PINN-proj, PINN-SC, PINN-KTT drift by 1e-2-1e-1 under random sampling. Same advantage in 2D (10^-6 vs 10^-3) and 3D (10^-7 vs 10^-4 to 10^-2). Solution `u` relative L2 error matches or exceeds baselines while conservation is exactly machine-precision on each minibatch.
