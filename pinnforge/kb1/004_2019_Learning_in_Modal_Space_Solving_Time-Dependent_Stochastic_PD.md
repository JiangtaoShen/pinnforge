---
slot: 4
title: "Learning in Modal Space: Solving Time-Dependent Stochastic PDEs Using Physics-Informed Neural Networks"
authors: [Dongkun Zhang, Ling Guo, George Em Karniadakis]
year: 2019
venue: "SIAM Journal on Scientific Computing (arXiv:1905.01205)"
gitrepo: ""
---

## TL;DR
Couple PINNs with the dynamically-orthogonal (DO) and bi-orthogonal (BO) modal decompositions to solve nonlinear time-dependent stochastic PDEs. Two parallel neural networks parameterise the spatial modes `u_i(x,t)` and the stochastic coefficients `Y_i(t; xi)`. The DO/BO constraints are written implicitly in the loss so the *explicit* evolution PDEs for the modes (which break down at eigenvalue crossings / singular covariance) are never needed.

## Problem
Classical DO requires `Cov(Y)` invertible; classical BO fails at eigenvalue crossings; both produce explicit evolution PDEs for each mode that are stiff and hard to integrate long-time. SPDEs with deterministic initial data immediately violate DO's invertibility assumption.

## Method
Truncated generalised KL expansion:
$$
u(x,t;\omega) = \bar{u}(x,t) + \sum_{i=1}^{N} u_i(x,t)\,Y_i(t;\xi(\omega))
$$
Parameterise with two sub-networks sharing parameters `theta`:
- `[bar_u, u_1, ..., u_N] = U_NN(x, t; theta)`
- `[Y_1, ..., Y_N] = Y_NN(t, xi; theta)`
where `xi` is a finite-dim random vector (gPC germ).

DO constraint: `<d u_i / dt, u_j> = 0`. BO constraint: `<u_i, u_j> = lambda_i delta_ij` and `E[Y_i Y_j] = delta_ij`. Write each constraint as a squared residual; total loss is sum of MSE terms, all evaluated at sample points by automatic differentiation:
$$
\mathcal{L} = \mathrm{MSE}_w + \mathrm{MSE}_{IC} + \mathrm{MSE}_{BC} + \mathrm{MSE}_{DO/BO} + \mathrm{MSE}_{reg}
$$
where MSE_w is the residual of the SPDE projected onto the mode basis (i.e. the weak/Galerkin residual):
$$
\mathrm{MSE}_w = \tfrac{1}{N_r}\!\sum\!\left|\partial_t u - \mathbb{E}_\xi\!\left[N_x[u]\right]\right|^2 + \cdots
$$
Inverse problems are obtained by adding noisy data MSE to L; computational cost is identical.

JAX sketch (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class ModalNet(nn.Module):
    """Outputs [bar_u, u_1..u_N] given (x,t)."""
    N: int
    h: int = 40
    depth: int = 4
    @nn.compact
    def __call__(self, xt):
        for _ in range(self.depth):
            xt = jnp.tanh(nn.Dense(self.h)(xt))
        return nn.Dense(self.N + 1)(xt)

class StochNet(nn.Module):
    """Outputs [Y_1..Y_N] given (t, xi)."""
    N: int
    d_xi: int
    h: int = 40
    depth: int = 4
    @nn.compact
    def __call__(self, t_xi):
        for _ in range(self.depth):
            t_xi = jnp.tanh(nn.Dense(self.h)(t_xi))
        return nn.Dense(self.N)(t_xi)

U = ModalNet(N=4); Y = StochNet(N=4, d_xi=4)
key = jax.random.PRNGKey(0)
params = {"U": U.init(key, jnp.zeros((1, 2))),
          "Y": Y.init(key, jnp.zeros((1, 1 + 4)))}

def loss_fn(params, xt_int, t_xi, x_ic, t_bc, x_bc):
    out_U  = U.apply(params["U"], xt_int)          # (Nr, N+1)
    bar_u, modes = out_U[:, 0:1], out_U[:, 1:]
    coeffs = Y.apply(params["Y"], t_xi)            # (Ns, N)
    # 1) weak residual: monte-carlo over xi for E[N_x[u]] then derivatives via jax.grad
    # 2) IC matching: u(x,0) = u0(x)
    # 3) BC matching
    # 4) DO/BO: < d_t u_i, u_j > = 0  (numerical inner product)
    # 5) E[Y_i]=0, Cov(Y)=I
    return MSE_w + MSE_IC + MSE_BC + MSE_DO + MSE_reg

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    grads = jax.grad(loss_fn)(params, *batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

For periodic BCs, replace input `x` by `(sin(2 pi x / L), cos(2 pi x / L))` to satisfy BCs exactly. Use `N = 3-5` modes; `d_xi = 3-10`; depth 4, width 40-50; Adam `lr=1e-3`.

## Results
On stochastic advection with deterministic IC (where DO fails outright), stochastic Burgers with many eigenvalue crossings (where BO fails), and 2-D reaction-diffusion, the NN-DO/BO method matches Monte-Carlo / gPC reference at ~1% mean and stdev error and successfully handles inverse problems with noisy initial data at the same cost as forward problems.
