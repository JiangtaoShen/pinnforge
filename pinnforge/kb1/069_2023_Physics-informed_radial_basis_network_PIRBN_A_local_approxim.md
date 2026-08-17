---
slot: 69
title: "Physics-informed radial basis network (PIRBN): A local approximating neural network for solving nonlinear PDEs"
authors: [Jinshuai Bai, Gui-Rong Liu, Ashish Gupta, Laith Alzubaidi, Xi-Qiao Feng, YuanTong Gu]
year: 2023
venue: Computer Methods in Applied Mechanics and Engineering 415, 116290
gitrepo: "https://github.com/JinshuaiBai/PIRBN"
doi: 10.1016/j.cma.2023.116290
---

## TL;DR
NTK analysis shows trained PINNs effectively become **local** approximators — yet when they fail on high-frequency / ill-posed PDEs, the issue is that their NTK never localizes. **PIRBN** hard-wires locality from the start: a single hidden layer of Gaussian radial basis units with **frozen centers** and only the output weights (and optionally widths) trainable. NTK theory proves convergence to a Gaussian process, and PIRBN beats deep PINN on high-frequency and ill-posed-domain problems with a single layer.

## Problem
PINNs need many layers and special tricks to fit high-frequency or singular solutions (e.g. `u(x) = sin(mu pi x)^2` with `mu` large, or domains with re-entrant corners). The NTK of these PINNs stays delocalized during training, so updates at one point spill globally and prevent the network from resolving fine features.

## Method
A PIRBN has the form
$$
u_\theta(x) = \frac{1}{\sqrt{d}}\sum_{i=1}^{d} a_i\,\vartheta_i(x),\qquad
\vartheta_i(x) = \exp\!\big(-b_i^2\,\|x - c_i\|^2\big)
$$
with:
- **Centers `c_i` frozen** on a regular grid covering the domain (and a bit beyond ill-posed regions). They are NOT trainable.
- Widths `b_i` either trainable or frozen at a value matched to inter-center spacing `h`: `b_i ~ 1 / (alpha h)`, `alpha ∈ [1, 2]`.
- Output weights `a_i ~ N(0, 1)` initialization (Le-Cun style for shallow nets).

Physics-informed loss is identical to PINN:
$$
\mathcal L(\theta) = \tfrac{1}{2}\!\sum_i |G[u_\theta(x_i^g)] - g(x_i^g)|^2 + \tfrac{1}{2}\!\sum_i |B[u_\theta(x_i^b)] - b(x_i^b)|^2
$$
PDE derivatives in `G[u]` are computed analytically through the Gaussian (closed form for any order) or by autograd. Locality is automatic because `vartheta_i(x)` is essentially zero outside `~ 3/b_i` of `c_i`.

Theorem (3.1.1, 3.1.3): as width `d -> infty`, trained PIRBN converges to a centered Gaussian process whose covariance is a deterministic kernel — training is a kernel regression in the chosen RBF feature space. Practical implication: PIRBN preserves the local approximation property during training, unlike PINN.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class PIRBN(nn.Module):
    centers: jnp.ndarray            # [d, n], frozen via sow / lax.stop_gradient
    b_init:  float = 1.0
    trainable_b: bool = False

    @nn.compact
    def __call__(self, x):
        c = jax.lax.stop_gradient(self.centers)            # frozen
        d = c.shape[0]
        if self.trainable_b:
            b = self.param("b", nn.initializers.constant(self.b_init), (d,))
        else:
            b = jnp.full((d,), self.b_init)
        a = self.param("a", nn.initializers.normal(stddev=1.0), (d,))
        diff = x[None, :] - c                              # [d, n]
        r2   = jnp.sum(diff**2, axis=-1)
        phi  = jnp.exp(-(b**2) * r2)
        return jnp.sum(phi * a) / jnp.sqrt(d)              # scalar

# 1-D example: u_xx - f = 0 with Gaussians on a uniform grid
N_c     = 200
centers = jnp.linspace(-1.0, 1.0, N_c).reshape(-1, 1)
net     = PIRBN(centers=centers, b_init=1.0 / (2.0 * (2.0 / N_c)),
                trainable_b=True)
params  = net.init(jax.random.PRNGKey(0), jnp.zeros((1,)))

def u(params, x):  return net.apply(params, x)             # scalar

def loss_fn(params, X_in, X_bd, f_fn, g_fn):
    def u_scalar(x): return u(params, x)
    u_xx = jax.vmap(lambda x: jax.grad(jax.grad(u_scalar))(x).squeeze())(X_in)
    L_g  = jnp.mean((u_xx - jax.vmap(f_fn)(X_in))**2)
    L_b  = jnp.mean((jax.vmap(u_scalar)(X_bd) - jax.vmap(g_fn)(X_bd))**2)
    return L_g + L_b

opt = optax.adam(1e-3); state = opt.init(params)

@jax.jit
def step(params, state, X_in, X_bd):
    grads = jax.grad(lambda p: loss_fn(p, X_in, X_bd, f, g))(params)
    upd, state = opt.update(grads, state, params)
    return optax.apply_updates(params, upd), state

for it in range(20000):
    params, state = step(params, state, X_in, X_bd)
```

Hyper-params: `d = 100..400`, Gaussian width `b` such that adjacent RBFs overlap (`b ~ 1/h`); Adam(1e-3); long training (1e4-1e5 iters). Compatible with all PINN add-ons (adaptive weighting, RAR sampling, domain decomp).

## Results
On `u_xx = -mu^2 pi^2 sin(mu pi x)` with `mu = 8` (high frequency) and on an L-shaped Poisson with a singular corner, single-layer PIRBN reaches relative L2 `~1e-3..1e-5` while a 4-layer PINN gets stuck at `~1e-1`. PIRBN training cost is lower (single layer, frozen centers) and the NTK stays diagonally dominant — locality is preserved throughout training.
