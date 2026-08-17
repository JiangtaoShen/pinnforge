---
slot: 124
title: "Robust Deep FOSLS for Transmission Problems"
authors: [Alejandro Duque-Salazar, Paulina Sepulveda, Carlos Uriarte, Jamie M. Taylor, David Pardo]
year: 2026
venue: arXiv:2604.17549
gitrepo: ""
---

## TL;DR
A First-Order System Least-Squares (FOSLS) deep solver for `-∇·(κ∇u)=f` with heterogeneous κ, where the L2 residual is preconditioned by the energy-norm Poincare constant `C_P^κ`. The resulting loss is provably norm-equivalent to the κ-weighted energy norm with constants independent of material contrast, has a passive variance-reduction property (gradient variance decreases as loss decreases), and uses ReQU activations to suppress quasi-Gibbs oscillations near discontinuities. Solver-in-the-loop: linear least-squares solves on the neural-network-spanned subspace, Adam updates the subspace itself.

## Problem
PINNs on transmission problems with discontinuous κ fail because residuals contain Dirac-deltas. VPINN/Deep-Ritz handle weak solutions but their stochastic-gradient variance does not decrease with the loss → unstable training under coarse quadrature. Standard Deep-FOSLS uses unweighted norm whose equivalence constants depend on κ — small loss does not imply small error under high material contrast (e.g. κ₀=10⁻⁶ or 10⁶).

## Method

### A. Robust weighted FOSLS
Let `q = −κ∇u`. The first-order system is `div q = f`, `q + κ∇u = 0`. With weighted energy norm
$$|||(u,q)|||_\kappa^2 = \|u\|_{H^1_{0,\kappa}}^2 + \|q\|_{(L^2_{1/\kappa})^d}^2 + (C_P^\kappa)^2\|\mathrm{div}\,q\|_{L^2}^2$$
the proposed loss is
$$L(u,q)=\|\kappa^{-1/2}q + \kappa^{1/2}\nabla u\|_{(L^2)^d}^2 + 2(C_P^\kappa)^2\|\mathrm{div}\,q - f\|_{L^2}^2$$
With `C = (C_P^κ)²`, `C_L = 2(C_P^κ)²` the equivalence constants `c_1 = 1/8`, `c_2 = 2` are κ-independent (Theorem 1, Corollary 1).

### B. Passive variance reduction (Theorem 2)
For an unbiased stratified MC quadrature `Q_N^P1[I_θ]`,
$$\mathrm{Var}\Bigl(\partial_\theta Q_N^{P1}[I_\theta]\Bigr) \le 4C_{\text{grad}}^2|\Omega|\,L(u_\theta,q_\theta)$$
i.e. variance is bounded by the loss itself — automatic variance reduction.

### C. Neural-network-induced reduced order
Shallow (or deep) ReQU network: `Φ_l = σ(W_l Φ_{l-1} + b_l)` with `σ(x) = max(0,x)²`. Span `V_θ^u = span{g_D · Φ_L^{(j)}}` (smooth cutoff `g_D` enforces Dirichlet), `V_θ^q = span{Φ_L^{(j)} e_k}`. Element `(u,q) = Σ c^u_i φ_i + Σ c^q_i τ_i`.

### D. Solver-in-the-loop
Each Adam step:
1. Assemble blocks `(H_uu, H_uq, H_qq)`, RHS `f_θ`, with `(C_P^κ)²` baked in. Diagonal-scale `c̃ = D_θ c` for stability.
2. Solve Tikhonov LSQ `c̃* = (H̃_θ + μI)^{-1} f̃_θ` (μ=1e-12).
3. Estimate `C_P^κ` every 100 iters via generalized eigenvalue `H_uu a = λ M a` → take `1/√λ_min`.
4. Backprop loss w.r.t. NN parameters θ; Adam step.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

def requ(x): return jax.nn.relu(x) ** 2

class FOSLSNet(nn.Module):
    n_hidden: int = 16
    depth: int = 1
    @nn.compact
    def __call__(self, x):                                    # x: (N, d)
        h = x
        for _ in range(self.depth):
            h = requ(nn.Dense(self.n_hidden)(h))
        return h                                              # features (N, n_hidden)

def gD_1d(x): return x * (1 - x)                              # vanishes at 0,1

def assemble_loss(params, X, kappa_fn, f_fn, CP):
    def features(x): return net.apply(params, x)              # x: (N, 1) -> (N, n)
    Phi = features(X)
    gD = gD_1d(X).reshape(-1, 1)
    phi = gD * Phi                                            # (N, n)  spans u
    tau = Phi                                                 # (N, n)  spans q (1D)
    # column-wise spatial grads
    def phi_col(x, j):
        return (gD_1d(x).squeeze() * net.apply(params, x[None]).squeeze())[j]
    grad_phi = jax.vmap(lambda x: jax.jacrev(
        lambda xi: gD_1d(xi).squeeze() * net.apply(params, xi[None]).squeeze())(x))(X).reshape(Phi.shape)
    div_tau  = jax.vmap(lambda x: jax.jacrev(
        lambda xi: net.apply(params, xi[None]).squeeze())(x))(X).reshape(Phi.shape)
    k = kappa_fn(X).squeeze(-1)
    H_uu = (k[:, None] * grad_phi).T @ grad_phi               # n x n
    H_uq = grad_phi.T @ tau
    H_qq = ((1.0 / k)[:, None] * tau).T @ tau \
         + 2 * CP**2 * (div_tau.T @ div_tau)
    f = f_fn(X).squeeze(-1)
    rhs_q = 2 * CP**2 * (div_tau.T @ f)
    H = jnp.block([[H_uu, H_uq], [H_uq.T, H_qq]])
    rhs = jnp.concatenate([jnp.zeros_like(rhs_q), rhs_q])
    D = jnp.sqrt(jnp.clip(jnp.diag(H), 1e-15, None))
    Dinv = 1.0 / D
    H_t = (Dinv[:, None] * H) * Dinv[None, :]
    c_t = jnp.linalg.solve(H_t + 1e-12 * jnp.eye(H_t.shape[0]), Dinv * rhs)
    c = Dinv * c_t
    return 0.5 * c @ H @ c - c @ rhs                          # quadratic value

net = FOSLSNet()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 1)))
opt = optax.adam(1e-4); opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, X, CP):
    loss, grads = jax.value_and_grad(assemble_loss)(params, X, kappa_fn, f_fn, CP)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss
```

Hyperparameters: 1 hidden layer x 16-32 ReQU neurons (deep variant: identity-init subsequent layers), stratified MC quadrature `Q_N^P1` with N=50-2000, Adam lr=1e-4, `μ=1e-12, α₁=1e-8, α₂=1e-10, ε=1e-15`.

## Results
On 1D Poisson with `κ ∈ {1, κ₀}` for `κ₀ ∈ {10⁻⁶, 10⁻³, 10³, 10⁶}`: standard Deep-FOSLS shows loss-error decoupling under high contrast; robust FOSLS keeps near-linear loss-error correlation across all contrasts. With Deep-Ritz, N=300 quadrature points cause loss overflow; robust FOSLS converges stably with N=50. ReQU networks visibly suppress quasi-Gibbs oscillations vs tanh.
