---
slot: 130
title: "When PINNs Go Wrong: Pseudo-Time Stepping Against Spurious Solutions"
authors: [Sifan Wang, Shawn Koohy, Yiping Lu, Paris Perdikaris]
year: 2026
venue: arXiv:2604.23528
gitrepo: "https://github.com/sifanexisted/jaxpi2"
---

## TL;DR
PINNs often converge to spurious solutions (e.g. zero plateau in lid-driven cavity, smeared shock in inviscid Burgers) that satisfy the empirical residual loss at finite collocation points. Pseudo-time stepping `(u_k − u_{k-1})/τ + R[u_k] = 0` combined with collocation-point resampling exposes hidden residual defects (amplifies them by `O(τ²/h³)` in transition layers of width h). The authors give a closed-form adaptive pseudo-time step `τ_k = ‖Δu_k‖/‖Δr_k‖` (a Barzilai-Borwein-style finite-difference surrogate for the inverse Jacobian magnitude).

## Problem
Theorem 2.1: for any finite collocation set X_int, there exists a smooth `u†(t,x) = α_h(t) u*(t,x)` with α_h ≡ 1 before `t_0` and ≡ 0 after `t_0+h`, that exactly zeros the empirical residual loss while being trivially zero on a large region. Spectral bias makes PINNs prefer such solutions. Standard tricks (better architectures, loss balancing, second-order optimizers) don't fix this.

## Method

### A. Pseudo-time-relaxed residual loss
$$L_{\text{pts}}(\theta;\theta_{k-1}) = \frac{1}{N_{\text{int}}}\sum_i\Bigl(\frac{u_\theta(x_i)-u_{\theta_{k-1}}(x_i)}{\tau} + R_{\text{int}}[u_\theta](x_i)\Bigr)^2$$
At each step `k`, snapshot `θ_{k-1}` (detached) and minimise w.r.t. `θ_k`. Combine with random resampling of `X_int^k` every iteration.

### B. Why it works (Theorem 2.5)
For a spurious solution `u†` with transition-layer width `h`, the *fresh-batch* empirical residual is `E[L_{int}^{new}(u†)] = O(h^{-1})`. After one pseudo-time update `u†,+ = u† − τ R[u†]`, it becomes `O(h^{-1} + τ²h^{-3})`. So pseudo-time stepping *amplifies the visible residual* of spurious solutions when paired with resampling, making them unstable.

### C. Adaptive step size (the practical contribution)
Linearising the iteration around an interior solution `R[u*]=0`, locally `e_k ≈ (I + τ J*)^{-1} e_{k-1}`. Stability needs `max_i |1/(1+τλ_i)| < 1`. As surrogate use Barzilai-Borwein finite differences:
$$\widehat\tau^k = \gamma^k\,\frac{\|\Delta u_k\|}{\|\Delta r_k\| + \varepsilon},\quad \Delta u_k = u_{\theta_k}(X^k) - u_{\theta_{k-1}}(X^k),\ \Delta r_k = R[u_{\theta_k}](X^k) - R[u_{\theta_{k-1}}](X^k)$$
EMA smoothing: `τ^k = (1-β)τ^{k-1} + β τ̂^k`. Cosine annealing of `γ^k` (from initial-loss to current-loss ratio, CosineDecay) progressively shrinks τ in later training, where an over-large τ would destabilise optimisation. Update `τ` every `m` iterations (e.g. 1,000).

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax, math
from dataclasses import dataclass

@dataclass
class AdaptivePTS:
    tau:  float = 1.0
    tau0: float = 1.0
    beta: float = 0.5
    eps:  float = 1e-8
    m:    int   = 1000
    L_init: float = None
    step: int = 0

    def cosine_decay(self, L_cur):
        if self.L_init is None: self.L_init = float(L_cur)
        ratio = max(float(L_cur) / (self.L_init + 1e-12), 1e-3)
        return 0.5 * (1 + math.cos(math.pi * (1 - ratio)))

    def update(self, du_norm, dr_norm, L_cur):
        self.step += 1
        if self.step % self.m != 0: return self.tau
        gamma = self.cosine_decay(L_cur)
        tau_hat = gamma * float(du_norm) / (float(dr_norm) + self.eps)
        self.tau = (1 - self.beta) * self.tau + self.beta * tau_hat
        return self.tau

def pts_loss(params, params_prev, X_int, X_bc, tau, lam_bc=10.0):
    u_now  = net.apply(params,      X_int)
    u_prev = jax.lax.stop_gradient(net.apply(params_prev, X_int))
    r_now  = pde_res_fn(params, X_int)
    L_pts = jnp.mean(((u_now - u_prev) / tau + r_now) ** 2)
    L_bc  = jnp.mean(bc_res_fn(params, X_bc) ** 2)
    return L_pts + lam_bc * L_bc, (u_now, r_now)

net = ...                                                      # backbone (PirateNet / MLP+Fourier)
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
params_prev = jax.tree_util.tree_map(jnp.copy, params)
opt = optax.adam(1e-3); opt_state = opt.init(params)
pts = AdaptivePTS()

@jax.jit
def grad_step(params, params_prev, opt_state, X_int, X_bc, tau):
    (loss, (u_now, r_now)), grads = jax.value_and_grad(pts_loss, has_aux=True)(
        params, params_prev, X_int, X_bc, tau)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss, u_now, r_now

for k in range(N_iter):
    X_int = sample_fresh()                                     # FRESH resample each step
    # before-update reference values
    u_prev_eval = net.apply(params_prev, X_int)
    r_prev_eval = pde_res_fn(params_prev, X_int)
    params, opt_state, loss, u_now, r_now = grad_step(
        params, params_prev, opt_state, X_int, X_bc, pts.tau)
    pts.update(jnp.linalg.norm(u_now - u_prev_eval),
               jnp.linalg.norm(r_now - r_prev_eval),
               loss)
    params_prev = jax.tree_util.tree_map(jnp.copy, params)
```

Hyperparameters: PirateNet backbone (RFF + adaptive residual blocks), SOAP optimizer (β1=0.9, β2=0.999) lr=1e-3, m=1,000 step-update frequency, EMA β=0.5, ε=1e-8. Resample collocation points every iteration.

## Results
Across 10 benchmarks (linear advection c=50, lid-driven cavity Re=5000, inviscid Burgers shock, chaotic dynamics, reaction-diffusion, high-Re flows): adaptive PTS consistently beats baseline PINN and best-tuned fixed-τ PTS. Fixed-τ training-loss can deceive — different τ values give similar loss but vastly different solution accuracy. Lid-driven cavity Re=5000: baseline PINN collapses to U≈const; PTS+resample recovers reference flow. Inviscid Burgers: only PTS+resample captures the shock.
