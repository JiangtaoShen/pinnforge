---
slot: 019
title: "NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations"
authors: [Xiaowei Jin, Shengze Cai, Hui Li, George Em Karniadakis]
year: 2020
venue: "Journal of Computational Physics (arXiv:2003.06496)"
gitrepo: ""
---

## TL;DR
NSFnets are PINNs that solve incompressible Navier-Stokes either in velocity-pressure (VP) or vorticity-velocity (VV) form, with no Poisson splitting and no pressure data. The paper also benchmarks two dynamic loss-weighting variants that auto-balance residual/BC/IC terms and lets vanilla PINNs reach turbulence at Re_tau ~ 1000.

## Problem
For Navier-Stokes, vanilla PINNs need correctly balanced loss weights (alpha, beta) on BC/IC vs residual, otherwise pressure is mis-recovered and turbulence cannot be sustained. Fixed manual tuning is problem-dependent and expensive; pressure splitting introduces extra Poisson solves.

## Method
A single tanh-MLP maps (t,x,y,z) to (u,v,w,p) for VP, or (u,v,w,wx,wy,wz) for VV. Loss is L = L_e + alpha*L_b + beta*L_i, where L_e is the squared residual of the chosen PDE form, L_b boundary MSE, L_i initial MSE. Pressure has no BC/IC data — it falls out of the divergence-free constraint via AD. The VV form embeds vorticity = curl(u) as an extra equation and adds Neumann conditions cleanly.

Dynamic weight update (per iteration k):
$$
\hat{\alpha}^{(k+1)} = \frac{\max_\theta |\nabla_\theta L_e|}{\overline{|\nabla_\theta \alpha^{(k)} L_b|}}, \quad
\hat{\beta}^{(k+1)} = \frac{\max_\theta |\nabla_\theta L_e|}{\overline{|\nabla_\theta \beta^{(k)} L_i|}}
$$
Alternative (DW2):
$$
\hat{\alpha}^{(k+1)} = \frac{\overline{|\nabla_\theta L_e|}}{\overline{|\nabla_\theta L_b|}}
$$
EMA: alpha = (1-lambda)*alpha + lambda*alpha_hat, with lambda = 0.1.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class NSFnet(nn.Module):
    width: int = 50
    depth: int = 4
    out_dim: int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.width)(x))
        return nn.Dense(self.out_dim)(x)

def vp_residual(params, apply_fn, txyz, Re):
    # scalar field accessors
    def comp(i):
        return lambda t: apply_fn(params, t)[i]
    u_fn = lambda t: apply_fn(params, t)[0]
    v_fn = lambda t: apply_fn(params, t)[1]
    w_fn = lambda t: apply_fn(params, t)[2]
    p_fn = lambda t: apply_fn(params, t)[3]

    def laplacian(f, t):
        H = jax.hessian(f)(t)         # (4,4)
        return H[1,1] + H[2,2] + H[3,3]

    def res_point(t):
        u, v, w, p = u_fn(t), v_fn(t), w_fn(t), p_fn(t)
        du = jax.grad(u_fn)(t); dv = jax.grad(v_fn)(t)
        dw = jax.grad(w_fn)(t); dp = jax.grad(p_fn)(t)
        ut, ux, uy, uz = du[0], du[1], du[2], du[3]
        vt, vx, vy, vz = dv[0], dv[1], dv[2], dv[3]
        wt, wx, wy, wz = dw[0], dw[1], dw[2], dw[3]
        lap_u = laplacian(u_fn, t); lap_v = laplacian(v_fn, t); lap_w = laplacian(w_fn, t)
        e1 = ut + u*ux + v*uy + w*uz + dp[1] - lap_u/Re
        e2 = vt + u*vx + v*vy + w*vz + dp[2] - lap_v/Re
        e3 = wt + u*wx + v*wy + w*wz + dp[3] - lap_w/Re
        e4 = ux + vy + wz
        return e1*e1 + e2*e2 + e3*e3 + e4*e4
    return jnp.mean(jax.vmap(res_point)(txyz))

def loss_bc(params, apply_fn, x_bc, u_bc):
    return jnp.mean((jax.vmap(lambda x: apply_fn(params, x))(x_bc) - u_bc)**2)
def loss_ic(params, apply_fn, x_ic, u_ic):
    return jnp.mean((jax.vmap(lambda x: apply_fn(params, x))(x_ic) - u_ic)**2)

def flat_abs(g):
    return jnp.concatenate([jnp.abs(x).ravel() for x in jax.tree_util.tree_leaves(g)])

net = NSFnet()
params = net.init(jax.random.PRNGKey(0), jnp.zeros(4))
apply_fn = net.apply
opt = optax.adam(1e-3); opt_state = opt.init(params)
alpha, beta, lam = 1.0, 1.0, 0.1

@jax.jit
def step(params, opt_state, alpha, beta, x_col, x_bc, u_bc, x_ic, u_ic, Re):
    g_e = jax.grad(lambda p: vp_residual(p, apply_fn, x_col, Re))(params)
    g_b = jax.grad(lambda p: loss_bc(p, apply_fn, x_bc, u_bc))(params)
    g_i = jax.grad(lambda p: loss_ic(p, apply_fn, x_ic, u_ic))(params)
    max_e  = jnp.max(flat_abs(g_e))
    mean_b = jnp.mean(flat_abs(g_b))
    mean_i = jnp.mean(flat_abs(g_i))
    alpha = (1-lam)*alpha + lam * (max_e / jnp.maximum(mean_b*alpha, 1e-12))
    beta  = (1-lam)*beta  + lam * (max_e / jnp.maximum(mean_i*beta , 1e-12))
    def total(p):
        return (vp_residual(p, apply_fn, x_col, Re)
                + alpha*loss_bc(p, apply_fn, x_bc, u_bc)
                + beta *loss_ic(p, apply_fn, x_ic, u_ic))
    grads = jax.grad(total)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, alpha, beta
```

Recommended: Xavier init, tanh, Adam then L-BFGS-B, NN size 4x50 to 10x100 depending on flow, alpha0=beta0=1, lambda=0.1.

## Results
On Kovasznay flow VP outperforms VV; on 2D cylinder wake and 3D Beltrami flow both forms work, with dynamic weights consistently best. First demonstration of PINN sustaining turbulent channel flow at Re_tau ~ 1000 on subdomains using DNS boundary data.
