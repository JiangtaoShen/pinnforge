---
slot: 033
title: "Characterizing possible failure modes in physics-informed neural networks"
authors: [Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, Michael W. Mahoney]
year: 2021
venue: "NeurIPS 2021 (arXiv:2109.01050)"
gitrepo: "https://github.com/a1k12/characterizing-pinns-failure-modes"
---

## TL;DR
Vanilla PINN fails on even simple convection (beta>~10), reaction (rho>~5) and reaction-diffusion PDEs not because of network capacity but because the PDE-residual soft-constraint creates an ill-conditioned loss landscape. Two fixes: (i) curriculum regularization — train at small coefficient first, warm-start the next; (ii) seq2seq — train one network per short time chunk instead of the whole space-time.

## Problem
For u_t + beta u_x = 0 on the periodic [0, 2*pi) with beta=30 a 4x50 tanh PINN trained with L-BFGS reaches ~90% relative error. The loss landscape becomes increasingly non-convex as beta or reaction coefficient rho grows; lowering lambda_F just trades accuracy for ease and still misses the physics.

## Method
A. **Curriculum regularization**: parameterize the PDE by a difficulty knob c (convection beta, reaction rho, ...). Train a sequence of PINNs at c_1 < c_2 < ... < c_K = c_target. For each c_i, initialize from the trained weights of c_{i-1}.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import jaxopt

class PINN(nn.Module):
    width: int = 50; depth: int = 4
    @nn.compact
    def __call__(self, xt):
        for _ in range(self.depth):
            xt = nn.tanh(nn.Dense(self.width)(xt))
        return nn.Dense(1)(xt)

def convection_residual_point(params, apply_fn, xt, beta):
    def u_fn(z): return apply_fn(params, z)[0]
    g = jax.grad(u_fn)(xt)
    return g[1] + beta * g[0]

def total_loss(params, apply_fn, beta, x_ic, u_ic, x_b_l, x_b_r, x_r):
    L_ic = jnp.mean((jax.vmap(lambda z: apply_fn(params, z)[0])(x_ic) - u_ic)**2)
    L_b  = jnp.mean((jax.vmap(lambda z: apply_fn(params, z)[0])(x_b_l)
                   - jax.vmap(lambda z: apply_fn(params, z)[0])(x_b_r))**2)
    r    = jax.vmap(lambda z: convection_residual_point(params, apply_fn, z, beta))(x_r)
    return L_ic + L_b + jnp.mean(r**2)

def train_one(params, apply_fn, beta, x_ic, u_ic, x_b_l, x_b_r, x_r, n_iter=5000):
    solver = jaxopt.LBFGS(
        fun=lambda p: total_loss(p, apply_fn, beta, x_ic, u_ic, x_b_l, x_b_r, x_r),
        linesearch="zoom", maxiter=n_iter)
    return solver.run(params)[0]

net = PINN()
params = net.init(jax.random.PRNGKey(0), jnp.zeros(2))
for beta in [1, 5, 10, 15, 20, 25, 30]:
    params = train_one(params, net.apply, beta, x_ic, u_ic, x_b_l, x_b_r, x_r)
```

B. **Sequence-to-sequence learning**: partition [0,T] into chunks of size Delta_t (~0.05-0.1). Train a separate PINN on chunk k with IC from the previous chunk's prediction at t = t_k.

```python
def step_seq2seq(beta, t_lo, t_hi, ic_fn, key, n_iter=5000):
    net_k  = PINN()
    params = net_k.init(key, jnp.zeros(2))
    x_ic_k = jnp.stack([x_grid, jnp.full_like(x_grid, t_lo)], axis=-1)
    u_ic_k = jax.vmap(ic_fn)(x_ic_k)
    x_r_k  = sample_collocation(t_lo, t_hi)
    x_b_l, x_b_r = sample_periodic_bc(t_lo, t_hi)
    return net_k, train_one(params, net_k.apply, beta,
                            x_ic_k, u_ic_k, x_b_l, x_b_r, x_r_k, n_iter)

nets, paramss = [], []
nets0, params0 = step_seq2seq(beta_target, 0.0, dt, true_ic, jax.random.PRNGKey(0))
nets.append(nets0); paramss.append(params0)
for k in range(1, K):
    t_lo, t_hi = k*dt, (k+1)*dt
    def prev_ic(x, prev_params=paramss[-1], prev_apply=nets[-1].apply, t_=t_lo):
        z = jnp.array([x[0], t_])
        return prev_apply(prev_params, z)[0]
    nk, pk = step_seq2seq(beta_target, t_lo, t_hi, prev_ic, jax.random.PRNGKey(k))
    nets.append(nk); paramss.append(pk)
```

Recommended: 4x50 tanh, L-BFGS with strong Wolfe line search, curriculum steps of ~5 between easier and target, Delta_t = 0.05-0.1 for seq2seq. Periodic BC implemented as MSE between left/right boundary outputs.

## Results
1D convection beta=30: vanilla PINN rel-err 0.90; curriculum 0.02 (~50x lower). beta=40: 0.96 -> 0.05. 1D reaction-diffusion (rho=5, nu=3): seq2seq with dt=0.05 cuts rel-err ~40x (almost two orders of magnitude) vs whole-space-time PINN. Both methods also smooth the loss landscape.
