---
slot: 80
title: "Physics-Informed Neural Networks for High-Frequency and Multi-Scale Problems using Transfer Learning"
authors: [Abdul Hannan Mustajab, Hao Lyu, Zarghaam Rizvi, Frank Wuttke]
year: 2024
venue: "Applied Sciences (arXiv:2401.02810)"
gitrepo: ""
---

## TL;DR
Vanilla PINNs fail on high-frequency PDEs due to spectral bias and the non-convex loss landscape. The fix is a frequency-ramping **transfer-learning** strategy: train a baseline at low frequency, then warm-start training at the next-higher frequency. Adam-trained baselines transfer far better than L-BFGS baselines (which over-fit local minima).

## Problem
For a damped harmonic oscillator with natural frequency `omega_0`, vanilla PINN with `100` collocation points and 5x64 MLP converges easily at 20 Hz, slowly at 30 Hz, and **fails** at 40 Hz with L-BFGS (loss stuck at 1e-1). The same spectral bias appears for the 1-D wave equation with multi-scale source.

## Method
Train one baseline `f_theta` at low frequency `omega_a` to loss `~1e-3`, then *fine-tune* the same architecture with the warm-started weights against the higher frequency `omega_b` for fewer iterations. Repeat the ramp (20 -> 30 -> 40 -> 50 -> 60 Hz). For the wave equation, the analogue is increasing wavenumber `k`. Use **temporal loss weighting**: the IC loss term `W_I L_I` is given the largest weight at the start of training so initial-condition residuals are minimised first.

Loss for the damped oscillator (under-damped, `delta = mu/(2m)`):
$$ r(t) = m\,\ddot{\hat u}_\theta + \mu\,\dot{\hat u}_\theta + k\,\hat u_\theta,\quad \mathcal{L}=W_F \|r\|^2 + W_I \big(|\hat u(0)-1|^2 + |\dot{\hat u}(0)|^2\big) $$

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class PINN(nn.Module):
    H: int = 64
    depth: int = 5
    @nn.compact
    def __call__(self, t):                          # t: (B, 1)
        for _ in range(self.depth):
            t = nn.tanh(nn.Dense(self.H)(t))
        return nn.Dense(1)(t)

def osc_loss(params, t_col, omega, mu, k, m=1.0, W_I=100., W_F=1.0):
    u_of = lambda ti: PINN().apply(params, ti[None])[0, 0]
    du   = lambda ti: jax.grad(u_of)(ti)
    ddu  = lambda ti: jax.grad(du)(ti)
    r    = jax.vmap(lambda ti: m*ddu(ti) + mu*du(ti) + k*u_of(ti))(t_col[:, 0])
    t0   = jnp.zeros(())
    L_F  = jnp.mean(r**2)
    L_I  = (u_of(t0) - 1.0)**2 + du(t0)**2
    return W_F*L_F + W_I*L_I

def train(params, opt_state_adam, omega, mu, k, n_adam=20000, n_lbfgs=2000):
    adam = optax.adam(1e-3)
    @jax.jit
    def adam_step(params, state):
        grads = jax.grad(osc_loss)(params, t_col, omega, mu, k)
        upd, state = adam.update(grads, state); return optax.apply_updates(params, upd), state
    for _ in range(n_adam):
        params, opt_state_adam = adam_step(params, opt_state_adam)
    # Switch to L-BFGS
    lbfgs = optax.lbfgs(learning_rate=0.1, memory_size=100,
                        linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=20))
    state_l = lbfgs.init(params)
    def lbfgs_step(params, state):
        v, g = jax.value_and_grad(osc_loss)(params, t_col, omega, mu, k)
        upd, state = lbfgs.update(g, state, params, value=v, grad=g,
                                  value_fn=lambda p: osc_loss(p, t_col, omega, mu, k))
        return optax.apply_updates(params, upd), state
    for _ in range(n_lbfgs):
        params, state_l = lbfgs_step(params, state_l)
    return params

# Transfer-learning ramp
omegas = [20.0, 30.0, 40.0, 50.0, 60.0]
params = PINN().init(jax.random.PRNGKey(0), jnp.zeros((1, 1)))
state_a = optax.adam(1e-3).init(params)
for w_target in omegas:
    omega = w_target; k = omega**2; mu = 2*0.05*omega
    params = train(params, state_a, omega, mu, k,
                   n_adam=20000 if w_target == omegas[0] else 2000)
```

Hyperparameters: 5 hidden layers x 64 (tanh), 100 collocation points, Adam `lr=1e-3` (~20k steps for the base, ~2k for each transfer), then L-BFGS with strong Wolfe at `lr=0.1`. **Use Adam-trained baselines for transfer**; L-BFGS baselines often trap fine-tuning. Temporal weighting `W_I = 100`, anneal to `1` over training.

## Results
Vanilla PINN at 40 Hz needs ~75k Adam steps and L-BFGS fails. With transfer learning from a 30 Hz Adam baseline, the 40 Hz problem converges to loss `~1e-3` in <2k epochs. Successive transfer 40 -> 50 -> 60 Hz keeps the network size fixed (4321 parameters) while solving all targets accurately. Similar gains on a multi-scale 1-D wave equation.
