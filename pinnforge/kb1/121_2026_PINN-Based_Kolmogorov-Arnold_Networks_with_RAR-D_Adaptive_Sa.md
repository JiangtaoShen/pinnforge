---
slot: 121
title: "PINN-Based Kolmogorov-Arnold Networks with RAR-D Adaptive Sampling for Solving Elliptic Interface Problems"
authors: [Zijuan Xin, Chenyao Wang, Feng Shi, Yizhong Sun]
year: 2026
venue: arXiv:2602.01876
gitrepo: ""
---

## TL;DR
Replace each MLP in a domain-decomposed dual-PINN for elliptic interface problems with a Kolmogorov-Arnold Network (KAN) — B-spline activations on edges — and combine with Residual-based Adaptive Refinement with Diversity (RAR-D). Achieves 2-10x lower L2 error than equal-budget MLP-PINNs with a much smaller network (3 neurons/layer vs 20).

## Problem
Elliptic interface problems `-∇·(a_i ∇u_i)=f_i` in Ω_i, with jump conditions `[u]=φ, [a∇u·n]=ψ` on Γ, exhibit non-smooth solutions and flux discontinuities near the interface. Vanilla MLP-PINNs need many neurons and uniformly sampled collocation points, but uniform sampling under-resolves interface-localised residuals.

## Method

### A. Dual KAN architecture
Decompose Ω = Ω_1 ∪ Ω_2 separated by Γ. Approximate u_i with an independent KAN per subdomain: `u_i(x) ≈ KAN_i(x) = (Φ^i_L ∘ ... ∘ Φ^i_1)(x)`. Each KAN layer applies univariate learnable functions on edges:
$$\phi(x)=c_r\,r(x)+c_B\sum_{i=1}^{G+m}c_iB_i(x),\qquad r(x)=\mathrm{SiLU}(x)$$
with m-th order B-splines `B_i` on a G-interval grid; `c_r, c_B, c_i` trainable.

### B. Composite loss
$$L=L_{\Omega_1}+L_{\Omega_2}+L_\Gamma+L_{\partial\Omega_1}+L_{\partial\Omega_2}$$
PDE residual in each subdomain, jump conditions `‖[u]−φ‖²+‖[a∇u·n]−ψ‖²` on Γ, Dirichlet on outer boundary. Latin Hypercube initial sampling.

### C. RAR-D adaptive sampling
Compute residual `ϖ(x) = |−∇·(a_i∇u_i)−f_i|`. Build resampling density:
$$p(x)\propto \frac{\varpi^k(x)}{\mathbb{E}[\varpi^k(x)]}+c$$
Recommended `k=2, c=0 or 1`. Re-draw the residual point set from `p(x)` every 1000-2000 Adam steps, after a 20000-step warm-up.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

def silu(x): return x * jax.nn.sigmoid(x)

def b_spline_basis(x, grid, m):
    # Cox-de Boor; x: (..., 1), grid: (G+2m+1,) → (..., G+m)
    bases = ((x[..., None] >= grid[:-1]) & (x[..., None] < grid[1:])).astype(x.dtype)
    for k in range(1, m + 1):
        left  = (x[..., None] - grid[:-(k+1)]) / (grid[k:-1]   - grid[:-(k+1)] + 1e-12) * bases[..., :-1]
        right = (grid[k+1:]   - x[..., None]) / (grid[k+1:]   - grid[1:-k]    + 1e-12) * bases[..., 1:]
        bases = left + right
    return bases                                                # (..., G+m)

class KANLinear(nn.Module):
    in_dim: int
    out_dim: int
    G: int = 10
    m: int = 3
    grid_lo: float = -1.0
    grid_hi: float = 1.0

    @nn.compact
    def __call__(self, x):                                       # x: (B, in_dim)
        h = (self.grid_hi - self.grid_lo) / self.G
        grid = jnp.linspace(self.grid_lo - self.m*h,
                            self.grid_hi + self.m*h, self.G + 2*self.m + 1)
        c_sp  = self.param("c_spline", lambda k: 0.1*jax.random.normal(
                            k, (self.out_dim, self.in_dim, self.G + self.m)))
        c_res = self.param("c_res", lambda k: jnp.ones((self.out_dim, self.in_dim)))
        # per-feature basis: (B, in_dim, G+m)
        B = jax.vmap(lambda xi: b_spline_basis(xi, grid, self.m), in_axes=-1, out_axes=1)(x)
        spline = jnp.einsum("bif,oif->bo", B, c_sp)
        res    = jnp.einsum("bi,oi->bo", silu(x), c_res)
        return spline + res

class DualKAN(nn.Module):
    layout: tuple = (2, 3, 3, 3, 1)
    G: int = 10
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.layout) - 1):
            x = KANLinear(self.layout[i], self.layout[i+1], G=self.G)(x)
        return x

def rard_resample(key, X, residual_fn, k=2, c=0.0, n_new=None):
    r = jnp.abs(residual_fn(X).squeeze()) ** k
    p = r / (r.mean() + 1e-12) + c
    p = p / p.sum()
    n_new = n_new or X.shape[0]
    idx = jax.random.choice(key, X.shape[0], shape=(n_new,), replace=True, p=p)
    return X[idx]
```

Hyperparameters: 3 layers x 3 neurons (KAN), G=5-15, Tanh-equivalent SiLU residual; vs MLP baseline 3 layers x 20. Sampling N1=200-300 (Ω1), N2=500 (Ω2), NΓ=300, N∂Ω=800. Adam 30-40k steps, RAR-D every 1-2k steps after 20k warm-up. k=2, c∈{0,1}.

## Results
Example 4.1 (Poisson with circular interface): KAN-A L2 1.03e-4 vs PINN 9.79e-4 (~10x better). Example 4.4 (annular star-shaped interface): KAN-A 8.52e-5 in Ω1 vs PINN 6.63e-4 (~8x). KAN uses ~3 neurons vs PINN's 20, and converges faster. RAR-D consistently helps both backbones.
