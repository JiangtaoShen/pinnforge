---
slot: 14
title: "Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems"
authors: [Ameya D. Jagtap, Ehsan Kharazmi, George Em Karniadakis]
year: 2020
venue: "Computer Methods in Applied Mechanics and Engineering"
gitrepo: ""
doi: "10.1016/j.cma.2020.113028"
---

## TL;DR
Conservative PINN (cPINN): decompose the domain into non-overlapping subdomains, deploy an independent PINN in each, and stitch them together at the interfaces by enforcing *strong-form flux continuity* (the conservation law itself) plus an *average-solution matching* term. Each subdomain uses its own architecture / activation / collocation density tuned to local solution regularity. Parallelisable; handles shocks and discontinuities (Burgers, KdV, compressible Euler).

## Problem
A single global PINN cannot accurately capture solutions with shocks, contact discontinuities, or steep gradients (compressible Euler), and offers no parallelism. Adding more collocation points everywhere is wasteful. Domain decomposition is needed but must preserve conservation.

## Method
Partition `Omega = U_p Omega_p`, `p = 1..N_sd`. In subdomain `Omega_p` use an MLP `u_{theta_p}` with locally adaptive activation `sigma(n a_k z)` (layer-wise slope `a_k`, scale `n=5`, init `a_k = 1`). The total loss is a sum of subdomain losses:
$$
\mathcal{L}(\tilde\Theta_p) = W_u^p\,\mathrm{MSE}_u^p \;+\; W_F^p\,\mathrm{MSE}_F^p \;+\; W_I^p\,\bigl(\mathrm{MSE}_{flux} + \mathrm{MSE}_{u_{avg}}\bigr) \;+\; S(a_k)
$$
Components:
- `MSE_F^p`: PDE residual `u_t + d/dx f(u, u_x; lambda)` at interior collocation in `Omega_p`.
- `MSE_u^p`: data / IC / BC residual.
- **Interface terms**, summed over interface points `x_I` on `dOmega_p cap dOmega_q`:
$$
\mathrm{MSE}_{flux} = \tfrac{1}{N_I}\sum |f(u_{\theta_p}) - f(u_{\theta_q})|^2,\qquad
\mathrm{MSE}_{u_{avg}} = \tfrac{1}{N_I}\sum \bigl|u_{\theta_p}(x_I) - \tfrac{1}{2}(u_{\theta_p}+u_{\theta_q})\bigr|^2
$$
The flux term is the strong-form conservation jump condition; for viscous flows `f` includes the diffusive term. The average-solution term is optional but speeds convergence.
- Slope-recovery regulariser `S(a) = 1 / [(1/(L-1)) sum_k exp(a_k)]` pushes activations to large slopes.

Each subdomain can independently choose: depth, width, activation, optimiser (Adam/L-BFGS), residual-point density. Networks are trained jointly; the interface gradients couple them.

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class AdaptiveMLP(nn.Module):
    h: int = 20
    depth: int = 4
    out_dim: int = 1
    n: float = 5.0
    @nn.compact
    def __call__(self, x):
        a = self.param("a", lambda key: jnp.full((self.depth,), 1.0 / self.n))
        for k in range(self.depth):
            x = jnp.tanh(self.n * a[k] * nn.Dense(self.h)(x))
        return nn.Dense(self.out_dim)(x)

# One sub-PINN per subdomain (each gets its own params pytree).
nets = [AdaptiveMLP() for _ in range(N_sd)]
keys = jax.random.split(jax.random.PRNGKey(0), N_sd)
params = [n.init(k, jnp.zeros((1, 2))) for n, k in zip(nets, keys)]
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

def flux(u, u_x, lam):                          # viscous Burgers
    return u ** 2 / 2 - lam * u_x

def cpinn_loss(params, batch, lam):
    L_tot = 0.0
    for p in range(N_sd):
        xt_r = batch["int"][p]
        def u_single(pp, t, x): return nets[p].apply(pp, jnp.array([[t, x]]))[0, 0]
        u   = jax.vmap(lambda t, x: u_single(params[p], t, x))(xt_r[:, 0], xt_r[:, 1])
        u_t = jax.vmap(lambda t, x: jax.grad(u_single, 1)(params[p], t, x))(xt_r[:, 0], xt_r[:, 1])
        u_x = jax.vmap(lambda t, x: jax.grad(u_single, 2)(params[p], t, x))(xt_r[:, 0], xt_r[:, 1])
        f_x = jax.vmap(lambda t, x: jax.grad(lambda tt, xx: flux(u_single(params[p], tt, xx),
                                                                  jax.grad(u_single, 2)(params[p], tt, xx), lam), 1)
                                    (t, x))(xt_r[:, 0], xt_r[:, 1])
        L_F = jnp.mean((u_t + f_x) ** 2)
        L_u = jnp.mean((nets[p].apply(params[p], batch["data"][p]) - batch["u"][p]) ** 2)
        L_tot = L_tot + W_F * L_F + W_u * L_u
    # Interface terms between each adjacent (p, q)
    for (p, q), xt_I in batch["interface"].items():
        up = nets[p].apply(params[p], xt_I)[:, 0]
        uq = nets[q].apply(params[q], xt_I)[:, 0]
        # ... flux & average-solution residuals via jax.grad as above
        L_tot = L_tot + W_I * (L_flux + L_avg)
    # slope-recovery (sum over all networks)
    for p in range(N_sd):
        a_p = params[p]["params"]["a"]
        L_tot = L_tot + 1.0 / jnp.mean(jnp.exp(a_p))
    return L_tot

@jax.jit
def train_step(params, opt_state, batch, lam):
    grads = jax.grad(cpinn_loss)(params, batch, lam)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

For inverse problems with piecewise-constant `lambda`, give each subdomain its own trainable `lambda_p`. Weights `W_u : W_F : W_I` typically 1 : 1 : 20 (interface emphasised). Use 2-16 subdomains.

## Results
On 1-D viscous Burgers, KdV, 2-D coupled Burgers, lid-driven cavity (Re=100, 1000), and compressible Euler with shocks, cPINN matches reference Riemann solutions with sharp shock capture (relative L2 ~ 1e-3 to 1e-4) where single-domain PINN smears shocks heavily. Inverse problems with piecewise viscosity solved to <1% in each region.
