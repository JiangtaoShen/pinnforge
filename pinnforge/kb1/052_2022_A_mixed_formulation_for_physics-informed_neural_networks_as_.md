---
slot: 052
title: "A mixed formulation for physics-informed neural networks as a potential solver for engineering problems in heterogeneous domains: comparison with finite element method"
authors: [Shahed Rezaei, A. Harandi, Ahmad Moeineddin, Bai-Xiang Xu, S. Reese]
year: 2022
venue: Computer Methods in Applied Mechanics and Engineering (arXiv:2206.13103)
gitrepo: ""
---

## TL;DR
For heterogeneous elasticity and Poisson problems, the paper proposes a PINN with one network per primary unknown (`u_x, u_y` or `T`) **and** one network per gradient/stress component (`σ_xx, σ_yy, σ_xy` or `q_x, q_y`). The loss combines the energy form (`L_EF`, on the primary), the strong-form residual on the stress outputs (`L_SF = div σ_o = 0`), a connection loss `L_cnc = σ_o − C ε(u)`, and Dirichlet/Neumann penalties. This keeps autograd order at one, matches FEM accuracy on heterogeneous microstructures, and handles sharp material interfaces.

## Problem
Vanilla displacement-PINN computes `div σ` via second-order autograd through `u`, which is brittle near material interfaces with sharp jumps in `E(x,y)` or `k(x,y)`. DEM avoids the second derivative but smooths concentrations. Mixed (u, σ)-output PINNs improve trainability but the residual loss still needs careful balancing.

## Method
**Five networks** (2D linear elasticity): `N_{u_x}, N_{u_y}, N_{σ_x}, N_{σ_y}, N_{σ_{xy}}`, each `(x,y) → ℝ`, tanh, ~3–4 hidden layers × 30. Or, for Poisson: `N_T, N_{q_x}, N_{q_y}`.

**Heterogeneous stiffness** stored as a callable `Ĉ(x,y)` (e.g., precomputed image of `E, ν` of the microstructure):
$$ \hat C(x,y) = \frac{E(x,y)}{(1-2\nu)(1+\nu)}\begin{bmatrix} 1-\nu & \nu & 0 \\ \nu & 1-\nu & 0 \\ 0 & 0 & (1-2\nu)/2 \end{bmatrix} $$

**Loss decomposition.**
$$ \mathcal{L} = \underbrace{\mathcal{L}_{EF} + \mathcal{L}_{DBC}}_{\text{on }u} \;+\; \underbrace{\mathcal{L}_{cnc} + \mathcal{L}_{SF} + \mathcal{L}_{NBC}}_{\text{on }\sigma_o} $$
$$ \mathcal{L}_{EF} = \mathrm{MAE}\!\Big(-\int_\Omega \tfrac{1}{2}\sigma:\varepsilon\,dV + \int_\Gamma (\sigma\cdot n)u\,dA\Big),\quad \mathcal{L}_{SF}=\mathrm{MSE}(\mathrm{div}\,\sigma_o),\quad \mathcal{L}_{cnc}=\mathrm{MSE}(\sigma_o-\hat C\hat\varepsilon(u)) $$
- `ε(u)` and `σ_o → div σ_o` need only one autograd pass each.
- Energy is approximated by Monte-Carlo integration over collocation points (or quadrature on a precomputed mesh).
- Use MAE for the global energy loss; MSE for the pointwise terms.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    hidden: int = 30
    depth:  int = 3
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x)

nets = {k: MLP() for k in ('ux', 'uy', 'sxx', 'syy', 'sxy')}

def C_tensor(xy):                                       # heterogeneous stiffness lookup
    E  = E_field(xy); nu = NU_field(xy)
    f  = E / ((1 - 2 * nu) * (1 + nu))
    return f, nu

def residuals(params, xy):
    def call(name, z): return nets[name].apply(params[name], z)
    # strain from u (Voigt: ε_xx, ε_yy, 2ε_xy)
    dux = jax.vmap(jax.grad(lambda z: call('ux', z[None]).sum()))(xy)
    duy = jax.vmap(jax.grad(lambda z: call('uy', z[None]).sum()))(xy)
    eps_xx = dux[:, 0:1]
    eps_yy = duy[:, 1:2]
    eps_xy = 0.5 * (dux[:, 1:2] + duy[:, 0:1])
    # constitutive σ from u
    f, nu = C_tensor(xy)
    s_xx_u = f * ((1 - nu) * eps_xx + nu * eps_yy)
    s_yy_u = f * (nu * eps_xx + (1 - nu) * eps_yy)
    s_xy_u = f * (1 - 2 * nu) * eps_xy
    # σ_o from independent networks
    s_xx = call('sxx', xy); s_yy = call('syy', xy); s_xy = call('sxy', xy)
    # divergence of σ_o (first-order autograd)
    ds_xx = jax.vmap(jax.grad(lambda z: call('sxx', z[None]).sum()))(xy)
    ds_yy = jax.vmap(jax.grad(lambda z: call('syy', z[None]).sum()))(xy)
    ds_xy = jax.vmap(jax.grad(lambda z: call('sxy', z[None]).sum()))(xy)
    div_x = ds_xx[:, 0:1] + ds_xy[:, 1:2]
    div_y = ds_xy[:, 0:1] + ds_yy[:, 1:2]
    L_SF  = jnp.mean(div_x ** 2 + div_y ** 2)
    L_cnc = jnp.mean((s_xx - s_xx_u) ** 2
                   + (s_yy - s_yy_u) ** 2
                   + (s_xy - s_xy_u) ** 2)
    energy = 0.5 * (s_xx * eps_xx + s_yy * eps_yy + 2 * s_xy * eps_xy)
    ux = call('ux', xy); uy = call('uy', xy)
    return L_SF, L_cnc, energy, (ux, uy), (s_xx, s_yy, s_xy)

def total_loss(params, xy_int, xy_bD, u_bD, xy_bN, t_bN):
    L_SF, L_cnc, w, (ux, uy), (sxx, syy, sxy) = residuals(params, xy_int)
    L_EF  = -jnp.mean(w) + boundary_work(xy_bN, ux, uy, t_bN)
    L_DBC = jnp.mean((ux[idx_D] - u_bD[:, 0:1]) ** 2
                   + (uy[idx_D] - u_bD[:, 1:2]) ** 2)
    L_NBC = jnp.mean((tractions(sxx, syy, sxy, n_bN) - t_bN) ** 2)
    return jnp.abs(L_EF) + L_DBC + L_cnc + L_SF + L_NBC

opt   = optax.adam(1e-3)
params = {k: nets[k].init(jax.random.PRNGKey(i), jnp.zeros((1, 2)))
          for i, k in enumerate(nets)}
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, xy_int, xy_bD, u_bD, xy_bN, t_bN):
    g = jax.grad(total_loss)(params, xy_int, xy_bD, u_bD, xy_bN, t_bN)
    upd, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state
```

Recommended hyperparameters: tanh MLPs, 3–4 layers × 30 per output; Adam lr=1e-3; SciANN-style separate networks per output; collocation = uniform grid on the microstructure image, with extra density at phase interfaces; equal weights typically work but tuned by gradient norm for stiff contrasts.

## Results
On 2-D plate with a circular inclusion (`E_inc/E_mat = 2`, ν=0.3, prescribed displacement) and on a 2-D heat conduction with sharp-edged inclusion, the mixed-formulation PINN matches FEM displacement/stress and temperature/flux fields, including at phase interfaces — whereas pure-displacement PINN smears the discontinuity. The only-first-order autograd loss is also significantly faster to train.
<!-- input was pymupdf-fallback plain text but content was clear -->
