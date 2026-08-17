---
slot: 045
title: "Physics-informed neural network simulation of multiphase poroelasticity using stress-split sequential training"
authors: [E. Haghighat, Daniel Amini, R. Juanes]
year: 2021
venue: Computer Methods in Applied Mechanics and Engineering (arXiv:2110.03049)
gitrepo: ""
---

## TL;DR
For coupled flow-mechanics in porous media, jointly training a PINN on all balance equations is unstable. The paper proposes: (i) cast governing equations in a specific non-dimensional form, (ii) introduce volumetric strain `ε_v` as an auxiliary network output with the kinematic constraint `ε_v - ∇·u = 0`, and (iii) train sequentially with the **fixed-stress-split** scheme (alternate mechanics and flow networks), which yields a much more stable PINN. The fixed-strain-split, in contrast, diverges — mirroring FEM stability results.

## Problem
A single PINN that minimises mass-balance + linear-momentum + constitutive + IC/BC losses tends either to the trivial null solution or drifts far from the true one; gradients of stress and pressure terms differ by orders of magnitude and the optimisation is non-convex.

## Method
**Three networks** with shared/independent parameters (SciANN-style):
- `u_θ_u(x,t)` for displacement
- `p_θ_p(x,t)` for pressure
- `ε_{v,θ_e}(x,t)` for volumetric strain (avoids second-derivative compositions of `u`)

**Non-dimensional balance equations** (single-phase, stress-split form):
$$ \nabla \bar\varepsilon_v + \nu_*\bar\nabla(\bar\nabla\cdot\bar{u}) + \tfrac{3}{2}\nu_*\bar\nabla\!\cdot(\bar\nabla\bar{u}) - b\bar\nabla\bar{p} + N_d\,d = 0 $$
$$ \frac{\partial \bar\sigma_v}{\partial \bar t} + \frac{\partial \bar p}{\partial \bar t} - \bar\nabla^2 \bar p + D_*\,\square = f^* \quad\text{(fixed-stress-split)} $$
with auxiliary `ε_v - ∇·u = 0` and constitutive `σ = K_drε_v 1 + 3ν_* K_dr e - b p 1`. The strain-split variant replaces `∂σ_v/∂t` by `∂ε_v/∂t` and is unstable.

**Sequential training (Algorithm).** At outer step `k`:
1. Freeze pressure net; train `(u_θ, ε_v)` on mechanics loss until tolerance.
2. Freeze mechanics nets; train pressure net on flow loss (with current `σ_v` cached) until tolerance.
3. Repeat until both losses stabilise.

**Loss decomposition** (mechanics block):
$$ \mathcal{L}_m = \lambda_1\|R_{\text{mom}}\|^2 + \lambda_2\|\varepsilon_v - \nabla\!\cdot u\|^2 + \lambda_3\|u - g_D\|^2_{\Gamma_D} + \lambda_4\|\sigma\!\cdot\!n - t_N\|^2_{\Gamma_N} $$
Weights `λ_i` set by gradient-normalisation (Wang et al.) restricted to each sub-block.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    d_out: int
    hidden: int = 20
    depth:  int = 5
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(self.d_out)(x)             # linear output

net_u  = MLP(d_out=2)        # (x,y,t) -> (u_x,u_y)
net_p  = MLP(d_out=1)
net_ev = MLP(d_out=1)        # volumetric strain as separate output

def mechanics_residuals(p_u, p_ev, p_p_detached, xy, t):
    xt = jnp.concatenate([xy, t], axis=-1)
    def u_of (z): return net_u .apply(p_u,  z)
    def ev_of(z): return net_ev.apply(p_ev, z)
    def p_of (z): return net_p .apply(p_p_detached, z)
    u    = u_of(xt); ev = ev_of(xt); p = p_of(xt)
    # ∇u  (B, 2, 3) row-wise jacobian
    du   = jax.vmap(jax.jacrev(u_of))(xt[:, None]).squeeze()
    div_u = du[:, 0, 0:1] + du[:, 1, 1:2]
    r_kin = ev - div_u
    grad_ev = jax.vmap(jax.grad(lambda z: ev_of(z[None]).sum()))(xt)
    grad_p  = jax.vmap(jax.grad(lambda z: p_of (z[None]).sum()))(xt)
    laplacian_u = sum(
        jax.vmap(jax.grad(lambda z, i=i: jax.grad(
            lambda zz, j=i: u_of(zz[None])[0, j])(z)[i]))(xt)
        for i in range(2))
    r_mom = grad_ev[:, :2] + 1.5 * nu_star * laplacian_u - b * grad_p[:, :2]
    return r_kin, r_mom

def flow_residual(p_p, p_u_detached, p_ev_detached, xy, t):
    xt = jnp.concatenate([xy, t], axis=-1)
    def p_of(z): return net_p.apply(p_p, z)
    p_t  = jax.vmap(jax.grad(lambda z: p_of(z[None]).sum()))(xt)[:, -1:]
    p_xx = sum(
        jax.vmap(jax.grad(lambda z, i=i: jax.grad(
            lambda zz: p_of(zz[None]).sum())(z)[i]))(xt)[:, i:i+1]
        for i in range(2))
    sv_t = sigma_v_t_from_frozen(p_u_detached, p_ev_detached, xt)
    return p_t + D_star * sv_t - p_xx                # fixed-stress-split

# alternating outer loop
opt_m = optax.adam(1e-3); opt_f = optax.adam(1e-3)
state_m = opt_m.init({'u': p_u, 'ev': p_ev})
state_f = opt_f.init(p_p)

for k in range(K_outer):
    p_p_fixed = jax.lax.stop_gradient(p_p)
    for _ in range(N_inner):                          # mechanics step
        def L_m(pm):
            r_kin, r_mom = mechanics_residuals(pm['u'], pm['ev'], p_p_fixed, xy, t)
            return jnp.mean(r_kin ** 2) + jnp.mean(r_mom ** 2) + bc_losses(pm)
        g = jax.grad(L_m)({'u': p_u, 'ev': p_ev})
        upd, state_m = opt_m.update(g, state_m)
        new = optax.apply_updates({'u': p_u, 'ev': p_ev}, upd)
        p_u, p_ev = new['u'], new['ev']
    p_u_fixed, p_ev_fixed = jax.lax.stop_gradient(p_u), jax.lax.stop_gradient(p_ev)
    for _ in range(N_inner):                          # flow step
        def L_f(pp):
            r = flow_residual(pp, p_u_fixed, p_ev_fixed, xy, t)
            return jnp.mean(r ** 2) + bc_p_losses(pp)
        g = jax.grad(L_f)(p_p)
        upd, state_f = opt_f.update(g, state_f)
        p_p = optax.apply_updates(p_p, upd)
```

Recommended: tanh MLPs ~5 layers × 20 units per field; Adam lr=1e-3; gradient-normalised loss weights within each sub-block; use Eq. `∂σ_v/∂t` (stress-split), avoid `∂ε_v/∂t` (strain-split). Always operate on dimensionless variables.

## Results
Validated on Mandel's consolidation, Barry-Mercer injection-production and a two-phase drainage problem. Sequential stress-split is stable and matches analytical/reference solutions; simultaneous training and strain-split sequential training are both unstable. The 3-network split (with explicit `ε_v` output) outperforms the displacement-only formulation.
