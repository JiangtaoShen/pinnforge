---
slot: 81
title: "Physics-informed neural networks with domain decomposition for the incompressible Navier-Stokes equations"
authors: [Linyan Gu, Shanlin Qin, Lei Xu, Rongliang Chen]
year: 2024
venue: "Physics of Fluids 36, 021914 (2024)"
gitrepo: ""
doi: "10.1063/5.0188830"
---

## TL;DR
NS-DDPINN splits the domain into non-overlapping sub-domains, fits an independent PINN per sub-domain, and stitches them with three new interface losses (residual continuity, vanishing residual gradient, flux continuity) on top of the standard `C0` average. Gradient pathologies are mitigated by a dynamic-weight rule (a la Wang et al.) and an attention/residual MLP backbone.

## Problem
Single-network PINNs degrade as the Navier-Stokes domain grows: optimisation gets stiffer, errors propagate from steep-gradient regions. Existing decomposition methods (XPINN/CPINN) only enforce `C0` and flux continuity, missing higher-order interface compatibility.

## Method
Decompose `Omega` into `{Omega_d}_{d=1}^{N_d}`. Per sub-domain `d`, an MLP `u_NN^d(x,t;Theta_d)` outputs `(u, v, w, p)`. The 3-D incompressible NS residuals are
$$ f_k = \partial_t u_k + k_1 (\mathbf{u}\cdot\nabla) u_k + \partial_{x_k} p - k_2 \nabla^2 u_k,\quad f_4 = \nabla\cdot\mathbf{u} $$
Sub-domain loss:
$$ \mathcal{L}_d = \mathcal{L}_d^{re} + W_d^{bc}\mathcal{L}_d^{bc} + W_d^{ini}\mathcal{L}_d^{ini} + W_d^{avg}\mathcal{L}_d^{avg} + W_d^{conti}(\mathcal{L}^{Ire}_d + \mathcal{L}^{Igrad}_d + \mathcal{L}^{Iflux}_d) $$
where `L^avg` is C0 on `u`, `L^Ire` matches PDE residuals, `L^Igrad` enforces `d_x' f = 0` (since `f = 0` everywhere ideally), and `L^Iflux` matches conservative fluxes across the interface.

**Gradient-pathology fixes**: (i) dynamic weights `W <- alpha W + (1-alpha) (max|grad L_re| / mean|W grad L_other|)`, alpha=0.9; (ii) modified-MLP backbone with two input encoders `f = sigma(W1 z0 + b1)`, `h = sigma(W2 z0 + b2)`, and gated layer update `h_i = z_i odot f + (1 - z_i) odot h`.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class ModMLP(nn.Module):                            # gated, 2 encoders
    H: int = 64
    depth: int = 8
    d_out: int = 4
    @nn.compact
    def __call__(self, x):
        f = nn.tanh(nn.Dense(self.H, name="U")(x))
        g = nn.tanh(nn.Dense(self.H, name="V")(x))
        z = nn.tanh(nn.Dense(self.H, name="L0")(x))
        for i in range(1, self.depth):
            h = z * f + (1 - z) * g
            z = nn.tanh(nn.Dense(self.H, name=f"L{i}")(h))
        return nn.Dense(self.d_out)(z)

def ns_residual_point(params_d, xi, k1, k2):
    u_of = lambda y: ModMLP().apply(params_d, y)
    grad_u = jax.jacrev(u_of)(xi)                   # (4, 4) wrt (x,y,z,t)
    H = jax.hessian(u_of)(xi)                       # (4, 4, 4)
    u, v, w, p = u_of(xi)
    ux, uy, uz, ut = grad_u[0]; vx, vy, vz, vt = grad_u[1]
    wx, wy, wz, wt = grad_u[2]; px, py, pz, _   = grad_u[3]
    lap_u = H[0,0,0] + H[0,1,1] + H[0,2,2]
    lap_v = H[1,0,0] + H[1,1,1] + H[1,2,2]
    lap_w = H[2,0,0] + H[2,1,1] + H[2,2,2]
    f1 = ut + k1*(u*ux + v*uy + w*uz) + px - k2*lap_u
    f2 = vt + k1*(u*vx + v*vy + w*vz) + py - k2*lap_v
    f3 = wt + k1*(u*wx + v*wy + w*wz) + pz - k2*lap_w
    f4 = ux + vy + wz
    return jnp.array([f1, f2, f3, f4])

def sub_loss(params_d, batch, neigh_params, k1, k2, W):
    L_re  = jnp.mean(jax.vmap(ns_residual_point, in_axes=(None,0,None,None))(
                     params_d, batch["x_col"], k1, k2)**2)
    L_bc  = bc_loss(params_d, batch["x_bc"])
    L_ini = ini_loss(params_d, batch["x_ini"])
    L_avg = sum(jnp.mean((ModMLP().apply(params_d, batch["x_int"])
                          - ModMLP().apply(pp, batch["x_int"]))**2) for pp in neigh_params)
    L_Ire = ...                                     # residual continuity
    L_Igrad = ...                                   # vanishing residual gradient
    L_Iflux = ...                                   # flux continuity
    return (L_re + W["bc"]*L_bc + W["ini"]*L_ini + W["avg"]*L_avg
            + W["con"]*(L_Ire + L_Igrad + L_Iflux))

opts   = [optax.adam(1e-3) for _ in range(Nd)]
states = [o.init(p) for o, p in zip(opts, params_list)]
alpha = 0.9; W = {"bc": 1.0, "ini": 1.0, "avg": 1.0, "con": 1.0}

@jax.jit
def step_d(params_d, state_d, batch, neigh_params, k1, k2, W, opt):
    grads = jax.grad(sub_loss)(params_d, batch, neigh_params, k1, k2, W)
    upd, state_d = opt.update(grads, state_d)
    return optax.apply_updates(params_d, upd), state_d

for it in range(N):
    for d in range(Nd):
        params_list[d], states[d] = step_d(
            params_list[d], states[d], batches[d], neighbours_of(d, params_list),
            k1, k2, W, opts[d])
    if it % 100 == 0:                               # dynamic re-weighting EMA
        W["bc"]  = alpha*W["bc"]  + (1-alpha)*ratio_metric_bc()
        W["ini"] = alpha*W["ini"] + (1-alpha)*ratio_metric_ini()
```

Hyperparameters: ModMLP with 8 hidden layers x 64; Adam (`lr=1e-3`) then L-BFGS-B fine-tune; Xavier init; per sub-domain `N_re ~ 1e4`, `N_int ~ 5e2`; alpha=0.9 for dynamic weight EMA. Pressure shift per sub-domain since `p` is up to constant.

## Results
On Kovasznay (2-D, Re=40), Beltrami (3-D), lid-driven cavity (Re=100), cylinder wake, and 3-D synthetic / real-vessel blood flow, NS-DDPINN with residual + flux + gradient continuity outperforms `C0`-only XPINN by 1-2 orders of magnitude in relative L2; dynamic weights + ModMLP add another factor of 2-5. Inverse-problem identification of `(k1, k2)` from sparse data is also stabilised across sub-domains by an extra parameter-continuity loss.
