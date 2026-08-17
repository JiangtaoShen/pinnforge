---
slot: 77
title: "Loss-attentional physics-informed neural networks"
authors: [Yanjie Song, He Wang, He Yang, Maria Luisa Taccari, Xiaohui Chen]
year: 2024
venue: "Journal of Computational Physics 501 (2024) 112781"
gitrepo: ""
doi: "10.1016/j.jcp.2024.112781"
---

## TL;DR
LA-PINN attaches one *loss-attentional network* (LAN) - a linear sub-net without activations - to each loss component (residual / IC / BC). The LAN takes the per-point squared-error vector and outputs a weighted sum, providing per-point learnable scale **and** bias; LAN parameters are trained adversarially (gradient ascent) against the main net, dynamically amplifying hard-to-fit points.

## Problem
Loss-component re-weighting (NTK / gradient-balancing) gives one weight per loss term, but hard-to-fit "stiff" collocation points within a single component need much larger gradients than easy ones. Point-weighting methods (SA-PINN) help, but typically learn only a scale per point and use a single network for the residual term.

## Method
For each loss component `j in {r, 0, b}`, define a *linear* LAN with parameters `xi_j = (W*_j, b*_j)` (no activations). Let `SE_j^i(theta) = |op_j[u_hat](x_i)|^2` be the squared error at point `i`. The weighted component loss is
$$ \mathcal{L}^{\star}_j(\theta,\xi_j) = \mathrm{MEAN}\big(W^{\star (l+1)}_j (\dots W^{\star (1)}_j [SE_j^i(\theta)]_i + b^{\star (1)}_j) + b^{\star (l+1)}_j\big) $$
which, because the LAN is purely linear, equals `(1/N_j) sum_i lambda_j^i(xi_j) SE_j^i(theta)` with per-point learnable `lambda_j^i`. Total loss `L* = L*_r + L*_0 + L*_b`.

Training is adversarial (min-max): the main net `theta` *minimises* L*, the LANs `xi_j` *maximise* it (gradient ascent on `xi`). This boosts the weight of points with large residuals (gradient w.r.t. `xi` is proportional to the SE itself), which in turn amplifies the back-prop signal at those points.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MainNet(nn.Module):
    H: int = 64
    depth: int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.H)(x))
        return nn.Dense(1)(x)

class LAN(nn.Module):
    """Purely linear: no activation. Maps SE vector (N,) -> scalar weighted-mean."""
    N: int
    hidden: int = 32
    layers: int = 2
    @nn.compact
    def __call__(self, se):                         # se: (N,)
        x = nn.Dense(self.hidden)(se)
        for _ in range(self.layers - 1):
            x = nn.Dense(self.hidden)(x)
        x = nn.Dense(self.N)(x)
        return x.mean()

def L_star(theta_params, xi_params, x_r, x_0, u0, x_b, u_b):
    SE_r = residual_se(theta_params, x_r)            # (N_r,)
    SE_0 = ic_se(theta_params, x_0, u0)
    SE_b = bc_se(theta_params, x_b, u_b)
    L_r = LAN(N=SE_r.size).apply(xi_params["r"], SE_r)
    L_0 = LAN(N=SE_0.size).apply(xi_params["0"], SE_0)
    L_b = LAN(N=SE_b.size).apply(xi_params["b"], SE_b)
    return L_r + L_0 + L_b

opt_theta = optax.adam(1e-3); state_theta = opt_theta.init(theta_params)
opt_xi    = optax.adam(1e-3); state_xi    = opt_xi.init(xi_params)

@jax.jit
def train_step(theta_params, xi_params, state_theta, state_xi, batch):
    # 1) maximise w.r.t. xi (gradient ascent => negate)
    g_xi = jax.grad(L_star, argnums=1)(theta_params, xi_params, *batch)
    g_xi = jax.tree_util.tree_map(lambda g: -g, g_xi)
    upd_xi, state_xi = opt_xi.update(g_xi, state_xi)
    xi_params = optax.apply_updates(xi_params, upd_xi)
    # 2) minimise w.r.t. theta
    g_th = jax.grad(L_star, argnums=0)(theta_params, xi_params, *batch)
    upd_th, state_theta = opt_theta.update(g_th, state_theta)
    theta_params = optax.apply_updates(theta_params, upd_th)
    return theta_params, xi_params, state_theta, state_xi
```

Hyperparameters: main net tanh MLP 4 x 50-100; LAN with `layers=2` hidden, hidden=16-64 (kept tiny); Adam `lr=1e-3` for both; same number of inner steps for `theta` and `xi`. Initialise LAN output layer so initial weights `lambda_j^i ~ 1`.

## Results
On Burgers, Allen-Cahn, KdV and 2-D NS problems, LA-PINN consistently beats vanilla PINN, SA-PINN and gradient-balanced PINN, often by 1-2 orders of magnitude in relative L2. Weight maps confirm LANs assign larger weights and steeper update directions to stiffness regions (shock fronts, sharp transitions).
