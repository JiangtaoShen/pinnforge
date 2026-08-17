---
slot: 104
title: "C-PINN: A neural network framework based on the Cordes condition for solving linear and fully nonlinear equations in non-divergence form"
authors: [Bingcheng Hu, Lixiang Jin, Zhaoxiang Li]
year: 2026
venue: arXiv:2604.25606
gitrepo: ""
---

## TL;DR
For PDEs in non-divergence form `A:D²u + b·∇u − cu = f`, naive PINN residuals are extremely ill-conditioned because the principal-part Hessian eigenvalues are dispersed. C-PINN multiplies the residual point-wise by `α(x)=tr(A)/tr(A²)`, which by the Cordès condition turns the operator into a small perturbation of the Laplacian, giving a strictly contractive (and hence strongly convex) residual loss.

## Problem
Standard PINNs target divergence-form / low-order operators. For non-divergence linear PDEs and fully nonlinear equations (Hamilton–Jacobi–Bellman, Monge–Ampère), direct minimisation of `‖A:D²u−f‖²` has a highly non-convex landscape with severe gradient pathologies caused by the anisotropic, ill-conditioned second-order coefficient matrix `A(x)`.

## Method

### A. Cordès-preconditioned loss (linear non-divergence)
For `A(x)` satisfying the Cordès condition with parameter `ε∈(0,1)`, choose the optimal point-wise scaling
$$
\lambda(x)=\frac{\operatorname{tr}(A(x))}{\operatorname{tr}(A(x)^2)+\delta},\qquad \delta>0 \text{ small.}
$$
Theorem: `‖Δu − λ A:D²u‖_{L²} ≤ √(1−ε)‖Δu‖_{L²}`, i.e. the preconditioned operator is a strict contraction of the Laplacian. The C-PINN loss is
$$
\mathcal L_{\text{Cordès}}(\theta)=\frac{1}{N_{\text{int}}}\sum_{i}\Big(\lambda(x_i)\big(A:D^2 u_\theta + b\!\cdot\!\nabla u_\theta - c u_\theta\big)(x_i)-\lambda(x_i)f(x_i)\Big)^2,
$$
combined with a boundary MSE: `L = w_int L_Cordès + w_bc L_bc`.

### B. Dual-loop extension to nonlinear PDEs
Outer loop performs (semi-smooth) Newton linearisation. For HJB, freeze `u^{(k)}`, pick active `α* = argmax_α (L_α u^{(k)}−f_α)`, then the inner PDE becomes the linear surrogate `A^{α*}:D²u^{(k+1)} + b^{α*}·∇u^{(k+1)} − c^{α*}u^{(k+1)} = f^{α*}` solved with the static C-PINN loss using `A^{α*}`. For Monge–Ampère (`det(D²u)=f`), use Jacobi's formula so `A^{(k)} = cof(D²u^{(k)})`, RHS `f̃^{(k)} = f − det(D²u^{(k)}) + A^{(k)}:D²u^{(k)}`. A global warm-up trains plain PINN once, then alternate outer linearisation / inner C-PINN.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    hidden: int = 64; depth: int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x).squeeze(-1)

net = MLP()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))

def u_apply(params, x): return net.apply(params, x)

def cordes_loss(params, x_int, A_fn, b_fn, c_fn, f_fn, delta=1e-8):
    def u_scalar(xv): return u_apply(params, xv[None]).squeeze()
    grad_u = jax.vmap(jax.grad(u_scalar))(x_int)
    H      = jax.vmap(jax.hessian(u_scalar))(x_int)            # (B, d, d)
    A      = A_fn(x_int)                                       # (B, d, d)
    AD     = jnp.sum(A * H, axis=(1, 2))                       # A : D^2 u
    u_vals = jax.vmap(u_scalar)(x_int)
    Lu     = AD + jnp.sum(b_fn(x_int) * grad_u, axis=-1) - c_fn(x_int) * u_vals
    trA    = jnp.trace(A, axis1=1, axis2=2)
    trA2   = jnp.trace(A @ A, axis1=1, axis2=2)
    lam    = trA / (trA2 + delta)
    return jnp.mean((lam * (Lu - f_fn(x_int)))**2)

optimizer = optax.adam(3e-4)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, x_int, x_bc, g_bc, A_fn, b_fn, c_fn, f_fn, w_bc):
    def total(p):
        return cordes_loss(p, x_int, A_fn, b_fn, c_fn, f_fn) \
             + w_bc * jnp.mean((u_apply(p, x_bc) - g_bc)**2)
    g = jax.grad(total)(params)
    updates, opt_state = optimizer.update(g, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

# outer loop for HJB
for k in range(K_outer):
    alpha_star = pick_active_branch(params, x_int)             # arg max
    A_fn, b_fn, c_fn, f_fn = freeze_coeffs(alpha_star)
    for _ in range(N_inner):
        params, opt_state = step(params, opt_state, x_int, x_bc, g_bc,
                                  A_fn, b_fn, c_fn, f_fn, w_bc)
```

Hyper-parameters: Adam, `lr=3e-4`; ~10k interior + 1k boundary points; `δ=1e-8`; 4–20k inner iters; MLP backbone (tanh).

## Results
On a 2-D diffusion-dominated non-divergence equation with anisotropic `A`, C-PINN reaches `‖u−u_θ‖_{L²}≈6.4e-4` vs PINN `3.1e-3` at 16k epochs, with comparable per-iteration cost (~26 ms). The σ_max-proxy (local Lipschitz of ∇L) is visibly smaller and the loss landscape (filter-normalised) is markedly flatter. Method generalises to HJB, Monge–Ampère and optimal-transport tasks including high-dimensional cases.
