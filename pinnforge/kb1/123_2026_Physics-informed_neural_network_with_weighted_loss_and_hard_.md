---
slot: 123
title: "Physics-informed neural network with weighted loss and hard constraints for hyperbolic conservation laws"
authors: [Mahshid Sadat Ghoreishi, Hamid Naderan]
year: 2026
venue: Scientific Reports
gitrepo: "https://github.com/szl-c/pinn_CDnozzle"
---

## TL;DR
WHC-PINN combines three features for compressible Euler flows with shocks: (1) a gradient-weighting factor `λ = 1/(ε_1(|∇·u|−∇·u) + 1)` that down-weights residuals in shock zones, (2) a soft global mass/momentum/energy conservation penalty over the spatio-temporal slab, and (3) hard-constraint enforcement of pressure boundary conditions via `P(x) = g(x) + ℓ(x) N(x;θ)`. Solves both Riemann initial-value and converging-diverging nozzle boundary-value problems with the conservative form of the Euler equations.

## Problem
Vanilla PINNs cannot capture shocks: the optimiser smears infinite gradients to lower the residual loss, conflicting with the underlying physics. Prior fixes work either for IVPs (Riemann) or BVPs (CD-nozzle), and the latter usually requires the non-conservative form, leading to non-physical negative velocities. WHC-PINN aims at a unified solver in conservative form for both regimes.

## Method

### A. Non-dimensional quasi-1D Euler system (conservative form)
$$\partial_t U + \partial_x F = J,\quad U = (\rho'A',\rho'A'u',\rho'(\tfrac{T'}{\gamma-1}+\tfrac{\gamma u'^2}{2})A')^\top$$

### B. Gradient weighting in PDE loss
$$\lambda(x,t)=\frac{1}{\varepsilon_1(|\nabla\!\cdot\!\vec u|-\nabla\!\cdot\!\vec u)+1},\qquad w_{\text{PDE}}=w'_{\text{PDE}}\lambda$$
`λ→1` in smooth flow, `λ→1/(2ε_1|∇·u|+1)` in compression shocks. `ε_1=0.01` (Burgers), `0.2` (Euler).

### C. Global conservation soft penalty
$$L_{\text{CONS}}=(\text{Mass}(t_2)\!-\!\text{Mass}(t_1)\!-\!BD_{\text{Mass}})^2+(\text{Mom}\ldots)+(\text{Ene}\ldots)$$
where `Mass(t_k) = (1/N) Σ ρ_k A_k` over collocation points, `BD_*` are inlet/outlet flux integrals.

### D. Hard Dirichlet boundary for pressure only
$$\hat P(x;\theta) = g(x) + \ell(x)N(x;\theta),\quad \ell(x)=(x)(L-x),\quad g(x)=\frac{P_{\text{out}}-P_{\text{in}}}{P_0 L}x+\frac{P_{\text{in}}}{P_0}$$
Constraining only pressure preserves loss balance; constraining all variables disrupts gradient flow.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class WHCPINN(nn.Module):
    hidden: int = 50
    depth: int = 6
    out_dim: int = 4
    P_in: float = 1.0
    P_out: float = 0.81017
    a: float = 0.0
    b: float = 2.25
    @nn.compact
    def __call__(self, x, t):
        h = jnp.concatenate([x, t], axis=-1)
        for _ in range(self.depth):
            h = jnp.tanh(nn.Dense(self.hidden)(h))
        Y = nn.Dense(self.out_dim)(h)
        rho, u, T, P_raw = Y[..., :1], Y[..., 1:2], Y[..., 2:3], Y[..., 3:4]
        ell = (x - self.a) * (self.b - x)
        g   = (self.P_out - self.P_in) / (self.b - self.a) * (x - self.a) + self.P_in
        P_hard = g + ell * P_raw                              # exact at x=a,b
        return rho, u, T, P_hard

def gradient_weight(div_u, eps1=0.2):
    return 1.0 / (eps1 * (jnp.abs(div_u) - div_u) + 1.0)

def fields(params, x, t):
    return net.apply(params, x, t)

def euler_residual(params, X, gamma_=1.4, eps1=0.2):           # X: (N, 2)
    x, t = X[:, 0:1], X[:, 1:2]
    def cons(xt):                                              # xt: (2,)
        xi, ti = xt[0:1], xt[1:2]
        rho, u, T, P = fields(params, xi[None], ti[None])
        A = 1 + 2.2 * (xi - 1.5) ** 2
        return jnp.stack([(rho*A).squeeze(),
                          (rho*A*u).squeeze(),
                          ((T/(gamma_-1) + 0.5*u*u) * rho * A + P*A).squeeze(),
                          u.squeeze()])
    jac = jax.vmap(jax.jacrev(cons))(X)                        # (N, 4, 2)
    mass_x, mass_t = jac[:, 0, 0], jac[:, 0, 1]
    mom_x,  mom_t  = jac[:, 1, 0], jac[:, 1, 1]
    ene_x,  ene_t  = jac[:, 2, 0], jac[:, 2, 1]
    div_u          = jac[:, 3, 0]
    res_mass = mass_t + mass_x
    res_mom  = mom_t  + mom_x
    res_ene  = ene_t  + ene_x
    lam = gradient_weight(div_u, eps1)
    return lam * (res_mass**2 + res_mom**2 + res_ene**2)

net = WHCPINN()
```

Hyperparameters: 3-4 hidden layers x 30-50 neurons, Tanh; Adam 10k-20k epochs then L-BFGS 1k-5k for shock refinement. Weights `w_IC=10`, `w_PDE'=10`, `w_BC=1`, `w_CONS=1` (Riemann) / no CONS (CD nozzle). Riemann: 100 IC + 20k interior points; CD: 100 IC + 100 BC + 10k interior.

## Results
Sod Riemann (t=0.2): MSE_ρ 2.78e-5 to 3.4e-5 — shock and contact discontinuity captured. CD nozzle with shock at x=1.87 (P_b=0.81017): MSE_ρ 3.4e-5, MSE_M 8.2e-6, MSE_T 2.4e-5, MSE_P 9.3e-5 after Adam+L-BFGS. Supersonic CD (P_b=0.07726): MSE_ρ 2.78e-5. Maintains physical velocity (vs non-physical negative velocity in non-conservative baseline of Liang et al.).
