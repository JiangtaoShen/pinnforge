---
slot: 93
title: "A Helicity-Conservative Domain-Decomposed Physics-Informed Neural Network for Incompressible Non-Newtonian Flow"
authors: [Zheng Lu, Young Ju Lee, Jiwei Jia, Ziqian Li]
year: 2026
venue: arXiv:2604.08002
gitrepo: ""
---

## TL;DR
For incompressible non-Newtonian flow in Lamb (rotational) form, output only (u, p_total) and derive vorticity by autograd `omega = curl u`. This eliminates the helicity-pollution terms that arise when omega is a separate network output. Train with FBPINN-style overlapping subdomains blended by super-Gaussian partition of unity, plus causal slab-by-slab time marching.

## Problem
The rotational Navier-Stokes / non-Newtonian momentum equation conserves fluid helicity `H_f = integral u . omega` in the inviscid limit. If a PINN outputs `omega` as an independent head and uses a soft consistency penalty `||omega - curl u||^2`, the projection error injects spurious `integral grad p . omega` and `integral u . d_t omega` terms into the helicity balance (Theorem 3 of the paper). Long-time training also stalls on large 3-D space-time domains.

## Method
Velocity-pressure-only network with vorticity by autograd; spatial FBPINN partition; causal time slabs.

A. Architecture. On time slab `s`, the global ansatz is a convex blend of K local nets with super-Gaussian windows
$$
\bar w_k(x)=\frac{w_k(x)}{\sum_j w_j(x)},\quad w_k(x)=\exp\!\Big(-\frac{\|x-c_k\|^4}{2\sigma^4}\Big)
$$
$$
(u_\Theta, p_\Theta)(t,x)=\sum_{k=1}^K \bar w_k(x)\,N_k(t,x;\theta_k^{(s)}),\quad
\omega_\Theta=\nabla\times u_\Theta,\;\bar p=p_\Theta+\tfrac12|u_\Theta|^2
$$

B. Residuals (Lamb form, f=0):
$$
R_{\text{mom}}=\partial_t u - u\times\omega + \nabla\bar p + \mathrm{Re}^{-1}\nabla\times\omega,\quad R_{\text{div}}=\nabla\cdot u
$$
$$
\mathcal L^{(s)} = \tfrac{1}{|X_{PDE}|}\sum(\alpha\|R_{\text{mom}}\|^2+\beta\|R_{\text{div}}\|^2) + \mathcal L_{BC}^{(s)} + \mathcal L_{IC}^{(s)}
$$
with `L_BC` enforcing `u x n = 0` and `bar p = 0`, and `L_IC` matching the previous slab.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class LocalNet(nn.Module):
    hidden: int = 60; depth: int = 9; out_dim: int = 4    # (u1,u2,u3,p_tot)
    @nn.compact
    def __call__(self, tx):
        h = tx
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.hidden)(h))
        return nn.Dense(self.out_dim)(h)

class FBNet(nn.Module):
    centers: jnp.ndarray            # [K,3]
    sigma: float
    hidden: int = 60; depth: int = 9
    @nn.compact
    def __call__(self, tx):
        x = tx[..., 1:4]
        d4 = jnp.sum((x[:, None, :] - self.centers)**4, axis=-1)
        w  = jnp.exp(-d4 / (2 * self.sigma**4))
        w  = w / w.sum(-1, keepdims=True)                           # [B,K]
        K = self.centers.shape[0]
        outs = jnp.stack([LocalNet(self.hidden, self.depth)(tx)
                          for _ in range(K)], axis=1)                # [B,K,4]
        return (w[..., None] * outs).sum(1)

def field_u_p(params, tx):
    out = net.apply(params, tx)
    return out[..., :3], out[..., 3:4]                   # u, p_tot

def curl_u(params, tx):
    def u_i(tx_, i):
        return net.apply(params, tx_[None])[0, i]
    # Build Jacobian of u w.r.t. (x,y,z) = tx[1:4]
    def u_of_x(x3, t):
        tx_ = jnp.concatenate([t[None], x3])
        return net.apply(params, tx_[None])[0, :3]
    J = jax.vmap(lambda tx_: jax.jacrev(lambda x3: u_of_x(x3, tx_[0]))(tx_[1:4]))(tx)
    # J has shape [B,3,3] with J[b,i,j] = d u_i / d x_j
    omega = jnp.stack([J[:, 2, 1] - J[:, 1, 2],
                       J[:, 0, 2] - J[:, 2, 0],
                       J[:, 1, 0] - J[:, 0, 1]], axis=-1)
    return omega, J

def residuals(params, tx, Re):
    u, p = field_u_p(params, tx)
    omega, Ju = curl_u(params, tx)
    # d_t u via jacrev on time slot
    def u_only(tx_): return net.apply(params, tx_[None])[0, :3]
    Jfull = jax.vmap(jax.jacrev(u_only))(tx)             # [B,3,4]
    dut   = Jfull[..., 0]                                # d u / d t
    gp    = jax.vmap(jax.grad(lambda tx_: net.apply(params, tx_[None])[0, 3]))(tx)[..., 1:4]
    # curl omega: need d omega / d x via second-order autograd
    def omega_only(tx_):
        u_, J_ = curl_u(params, tx_[None])
        return u_[0]
    Jo = jax.vmap(jax.jacrev(omega_only))(tx)            # [B,3,4]
    cco = jnp.stack([Jo[:, 2, 2] - Jo[:, 1, 3],
                     Jo[:, 0, 3] - Jo[:, 2, 1],
                     Jo[:, 1, 1] - Jo[:, 0, 2]], axis=-1)
    R_mom = dut - jnp.cross(u, omega) + gp + (1.0/Re) * cco
    R_div = Ju[:, 0, 1] + Ju[:, 1, 2] + Ju[:, 2, 3]
    return R_mom, R_div

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, X_pde, X_bc, X_ic, u_prev, Re):
    def total(p):
        Rm, Rd = residuals(p, X_pde, Re)
        return (jnp.mean(Rm**2) + jnp.mean(Rd**2)
                + L_bc(p, X_bc) + L_ic(p, X_ic, u_prev))
    g = jax.grad(total)(params)
    updates, opt_state = optimizer.update(g, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

for s in range(N_seq):                                   # 100 slabs of dt=1e-2
    for it in range(1000):
        params, opt_state = step(params, opt_state, sample_pde(s),
                                  sample_bc(s), sample_ic(s), u_prev, Re)
    u_prev = jax.lax.stop_gradient(net.apply(params, slab_end_pts))
```

Hyperparameters: 9 hidden layers x 60 neurons (tanh) per local subnet, Adam, 100 slabs * 1000 iters, alpha=beta=1, super-Gaussian sigma chosen so neighboring windows overlap ~50%.

## Results
Manufactured 3-D solution on `[0,1]^3` to T=1: final L2 errors u 1.6e-3, omega 2.1e-2, p 3.5e-4; final loss median 1.16e-5 over 100 slabs. Helicity defect stays below 4.1e-6 and energy defect below 8.1e-7; the direct-vorticity baseline shows non-negligible `integral grad p . omega` pollution as predicted by Theorem 3.
