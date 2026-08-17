---
slot: 95
title: "A Theory-guided Weighted L2 Loss for solving the BGK model via Physics-informed neural networks"
authors: [Gyounghun Ko, Sung-jun Son, Seung Yeon Cho, Myeong-Su Lee]
year: 2026
venue: arXiv:2604.04971
gitrepo: ""
---

## TL;DR
For the BGK kinetic equation, an `O(eps^2)` standard PINN loss can coexist with `O(1)` solution error because errors concentrated in the high-velocity tail still corrupt the macroscopic moments. Multiply every L2 residual by a velocity weight `w(v) = 1 + alpha |v|^beta` with `beta > 7/2` to suppress tail errors; a stability theorem then guarantees `L_w-PINN -> 0` implies `||f - f_tilde||_2 -> 0`.

## Problem
The BGK equation `d_t f + v . grad_x f = (1/Kn)(M[f] - f)` has a Maxwellian relaxation kernel `M[f] = rho/(2*pi*T)^{3/2} exp(-|v-u|^2/(2T))` parameterized by moments `(rho, u, T)` that are velocity integrals of `f` weighted by `1, v, |v|^2`. A bump `K_eps(v)` placed in the high-`|v|` tail can have `||K_eps||_2 = O(eps)` yet contribute `O(1)` energy, so the standard L2 PINN loss `L_PINN = O(eps^2)` does not control the moments and the trained distribution relaxes to the wrong equilibrium.

## Method
Weight every residual MSE in velocity space by `w(v) >= 1`. With residuals
$$
R_{\text{pde}} = \partial_t f + v\cdot\nabla_x f - \tfrac1{\mathrm{Kn}}(M[f]-f),\;\;
R_{\text{ini}} = f(0,x,v)-f_0,\;\;
R_{\text{bc},i} = f|_{x_i=1}-f|_{x_i=0}
$$
$$
\mathcal L_{w\text{-PINN}} = \|w\,R_{\text{pde}}\|_2^2 + \lambda_{ini}\|w\,R_{\text{ini}}\|_2^2 + \lambda_{bc}\sum_i\|v_i w^2 R_{\text{bc},i}\|_2^2
$$
Practical choice: `w(v) = 1 + alpha |v|^beta` with `alpha > 0`, `beta > 7/2`. Theorem 5 (stability) under integrability `int (1+|v|^2)^2/w(v)^2 dv < inf` and `int w(v)^2 exp(-2 c_M |v|^2) dv < inf` gives
$$
\|w(f-\tilde f)(t)\|_2^2 \le C^*\Big(\|wR_{\text{ini}}\|_2^2 + \int_0^t\|wR_{\text{pde}}(s)\|_2^2 ds + \sum_i\int_0^t\|v_i w^2 R_{\text{bc},i}\|^2_{\partial_i} ds\Big)
$$

Network ansatz uses a micro-macro decomposition so `f_theta in X_M`:
$$
f_\theta(t,x,v) = M_{\tilde\rho_\theta, \tilde u_\theta, \tilde T_\theta}(v) + e^{-|v-\mu|^2/\tau}\,\tilde f^{\text{neq}}_\theta(t,x,v)
$$
where `(rho_tilde, u_tilde, T_tilde, f_neq_tilde)` are MLP outputs. The Maxwellian envelope guarantees fast tail decay so polynomial `w(v)` is integrable against `f_theta`.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

softplus = jax.nn.softplus

class BGKNet(nn.Module):                       # micro-macro ansatz
    hidden: int = 128; depth: int = 5
    @nn.compact
    def __call__(self, txv):
        t, x, v = txv[..., :1], txv[..., 1:4], txv[..., 4:7]
        h = txv
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.hidden)(h))
        macro = nn.Dense(5)(h)
        rho = softplus(macro[..., 0:1])
        u   = macro[..., 1:4]
        T   = softplus(macro[..., 4:5]) + 1e-3
        M   = rho/((2*jnp.pi*T)**1.5) * jnp.exp(-jnp.sum((v - u)**2, axis=-1, keepdims=True)/(2*T))
        env = jnp.exp(-jnp.sum(v**2, axis=-1, keepdims=True))
        neq = nn.Dense(1)(h)
        return M + env * neq

def w_fn(v, alpha=1.0, beta=4.0):
    return 1.0 + alpha * (jnp.linalg.norm(v, axis=-1, keepdims=True))**beta

net = BGKNet()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 7)))

def f_apply(params, txv): return net.apply(params, txv)

def pde_residual(params, txv, Kn):
    def f_scalar(txv_): return f_apply(params, txv_[None])[0, 0]
    grads = jax.vmap(jax.grad(f_scalar))(txv)               # [B,7]
    f_t = grads[..., 0:1]
    f_x = grads[..., 1:4]
    v   = txv[..., 4:7]
    rho, u, T = moments_quadrature(params, txv[..., :4])    # numerical integration over v
    M = (rho/((2*jnp.pi*T)**1.5)) * jnp.exp(-jnp.sum((v - u)**2, axis=-1, keepdims=True)/(2*T))
    f = f_apply(params, txv)
    return f_t + jnp.sum(v * f_x, axis=-1, keepdims=True) - (M - f) / Kn

def L_w_PINN(params, X_pde, X_ini, X_bc, Kn, lam_ini=1.0, lam_bc=1.0):
    w_pde = w_fn(X_pde[..., 4:7])
    R_pde = pde_residual(params, X_pde, Kn)
    L_pde = jnp.mean((w_pde * R_pde)**2)
    w_ini = w_fn(X_ini[..., 4:7])
    L_ini = jnp.mean((w_ini * (f_apply(params, X_ini) - f0(X_ini)))**2)
    L_bc = 0.0
    for i, (Xlo, Xhi) in enumerate(X_bc):
        wi = (w_fn(Xlo[..., 4:7])**2) * jnp.abs(Xlo[..., 4+i:5+i])
        L_bc = L_bc + jnp.mean((wi * (f_apply(params, Xhi) - f_apply(params, Xlo)))**2)
    return L_pde + lam_ini * L_ini + lam_bc * L_bc

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, X_pde, X_ini, X_bc, Kn):
    g = jax.grad(L_w_PINN)(params, X_pde, X_ini, X_bc, Kn)
    updates, opt_state = optimizer.update(g, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

Hyperparameters: alpha=1, beta=4 (> 7/2), Kn=1, ~5-layer MLPs with tanh + softplus heads, Adam, velocity quadrature on a Hermite-Gauss grid for moments and tail decay enforced by the Maxwellian envelope.

## Results
Two analytical counterexamples (small initial residual `f_eps^(1)` and small PDE residual `f_eps^(2)`) confirm standard L2 PINN loss can be `O(eps^2)` with `O(1)` error. Benchmark tests with the weighted loss show consistently higher solution accuracy and correct moments across BGK problems compared to the unweighted baseline; the stability bound is tight up to `C^*`.
