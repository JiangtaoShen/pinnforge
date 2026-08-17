---
slot: 110
title: "Deflation-PINNs: Learning Multiple Solutions for PDEs and Landau-de Gennes"
authors: [Sean Disarò, Ruma Rani Maity, Aras Bacho]
year: 2026
venue: arXiv:2603.27936
gitrepo: "https://github.com/SeanDisaro/DeflationPINNs"
---

## TL;DR
A single network simultaneously approximates `K` distinct solutions of a multi-solution PDE by combining a shared trunk net `τ(x)∈R^p` with `K` trainable branch weights `β^k∈R^p` (simplified DeepONet) and adding a *deflation* hinge loss that forces pairwise L²-separation between branches by a user-set radius `d_min`. Hard Dirichlet boundary constraints via star-domain radial extrapolation reduce the loss to a single PDE residual term.

## Problem
Vanilla PINNs find one PDE solution per random init; running many parallel PINNs (Zou et al.) gives no guarantee of distinctness and wastes capacity. Nonlinear PDEs like the Landau-de Gennes model of nematic liquid crystals have 6 stable solutions on a square (D1, D2, R1-R4). A scalable framework that finds *all known* equilibria in one training run is missing.

## Method

### A. Architecture (degenerated DeepONet)
Let `τ_1,...,τ_p : R^d → R` be outputs of a single shared trunk net. For each solution `k=1,...,K` store a trainable vector `β^k ∈ R^p`:
$$
\tilde G(k,x)=\sum_{i=1}^p \tau_i(x)\,\beta_i^k\approx u_k(x).
$$
Only `K·p` extra parameters vs. full DeepONet's `O(S·p)` sensor encoder. Universal approximation holds (special case of Chen-Chen).

### B. Hard Dirichlet via radial extrapolation on star domains
With star centre `x_0`, radial-boundary-distance `r_Ω(x)=‖x_b-x_0‖` to the boundary point along the ray:
$$
\tilde u_k(x)=h\!\big(\|x-x_0\|/r_\Omega(x)\big)\,\tilde f(x)+\big(1-h(\|x-x_0\|/r_\Omega(x))\big)\,\tilde G(k,x)
$$
with `h(0)=1, h(1)=0`; angular boundary lift `\tilde f(x)=ψ(φ_x,θ_x)`. This satisfies `\tilde u_k|_{∂Ω}=ψ` identically; removes boundary loss term.

### C. Deflation loss (pairwise hinge in L²)
$$
\mathcal L_{\text{Def}}=\frac{2}{K(K-1)}\sum_{i<j}\max\Big(1-\tfrac{1}{d_{\min}}\|\tilde G(i,\cdot)-\tilde G(j,\cdot)\|_{L^2},\,0\Big)
$$
$$
\mathcal L_{\text{total}}=\alpha\sum_{k=1}^K \mathcal E_G\big(\tilde G(k,\cdot)\big)+\beta\,\mathcal L_{\text{Def}},
$$
where `E_G` is the per-branch PDE residual MSE. Setting `L_Def=0` is equivalent to all branches being pairwise at least `d_min` apart.

```python
import jax, jax.numpy as jnp
import flax.linen as nn

class DeflationPINN(nn.Module):
    p: int = 64
    K: int = 6
    hidden: int = 64
    depth: int = 4

    @nn.compact
    def __call__(self, x, omega_fn, lift_fn):           # x: (N, d)
        h = x
        for _ in range(self.depth - 1):
            h = nn.tanh(nn.Dense(self.hidden)(h))
        tau = nn.Dense(self.p, name="trunk_out")(h)     # (N, p)
        beta = self.param("beta",
                          lambda k: 0.1*jax.random.normal(k, (self.K, self.p)))
        G   = tau @ beta.T                              # (N, K)
        w   = omega_fn(x)[:, None]                      # zero on dOmega
        f   = lift_fn(x)[:, None]                       # equals psi on dOmega
        return f + w * G                                # hard Dirichlet, (N, K)

def deflation_loss(params, apply_fn, x_col, omega_fn, lift_fn, d_min=0.5):
    # raw branch values (without hard-constraint multiplier) for L^2 separation
    tau = apply_fn(params, x_col, omega_fn, lift_fn, method="trunk")  # (N, p)
    beta = params["params"]["beta"]
    G = tau @ beta.T                                    # (N, K)
    K = G.shape[1]
    loss = 0.0
    for i in range(K):
        for j in range(i+1, K):
            L2 = jnp.sqrt(jnp.mean((G[:, i] - G[:, j])**2))
            loss = loss + jnp.maximum(1.0 - L2 / d_min, 0.0)
    return 2 * loss / (K * (K - 1))

def total_loss(params, apply_fn, x_col, pde_residual, omega_fn, lift_fn,
               d_min=0.5, alpha=1.0, beta=10.0):
    U = apply_fn(params, x_col, omega_fn, lift_fn)      # (N, K)
    pde = sum(jnp.mean(pde_residual(U[:, k], x_col)**2)
              for k in range(U.shape[1]))
    return alpha * pde + beta * deflation_loss(params, apply_fn, x_col,
                                               omega_fn, lift_fn, d_min)
```

Hyper-parameters: `p≈32–128`, `K`=expected number of solutions (6 for 2-D LdG), trunk MLP 4×64 tanh, `d_min` from FEM-known L²-gap (≈0.3-0.5 in LdG benchmarks), `optax.adam(1e-3)`, `α=1, β∈[1,100]`. For Landau-de Gennes: `Ω=[0,1]²`, `ε=0.02`, trapezoidal Dirichlet `Q_b`.

## Results
On the 2-D Landau-de Gennes problem with `ε=0.02`, a single Deflation-PINN with `K=6` recovers all six FEM-known stable equilibria (D1, D2, R1-R4) in one training run, matching FEM energies and director-field patterns. Convergence of branch weights `β^k` is monitored; deflation hinge naturally turns off once `d_min` is satisfied so optimisation focuses on PDE residuals. Code at github.com/SeanDisaro/DeflationPINNs.
