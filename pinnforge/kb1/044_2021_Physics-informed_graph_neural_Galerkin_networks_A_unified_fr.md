---
slot: 044
title: "Physics-informed graph neural Galerkin networks: A unified framework for solving PDE-governed forward and inverse problems"
authors: [Han Gao, Matthew J. Zahr, Jian-Xun Wang]
year: 2021
venue: Computer Methods in Applied Mechanics and Engineering (arXiv:2107.12146)
gitrepo: ""
---

## TL;DR
PI-GGN replaces the MLP of a continuous PINN with a Chebyshev graph-convolutional network that outputs nodal values on an unstructured FE mesh. The loss is the weak (continuous-Galerkin) residual of the PDE, integrated by Gaussian quadrature; essential BCs are imposed by static condensation, so no soft penalties are needed. The same network handles forward and inverse problems on arbitrary geometries.

## Problem
FC-PINNs have an infinite trial space, point-wise autodiff, and soft BC penalties that fail on complex geometries. CNN-based discrete PINNs are limited to Cartesian grids. Variational PINNs with MLP trial functions suffer "variational crimes" because quadrature on a black-box network is inaccurate.

## Method
**Mesh as graph.** Each FE node = graph node with input feature = coordinates; output feature = nodal solution `Û(χ;Θ) ≈ U`. Edges from element connectivity; build adjacency `A`, degree `D`, normalised Laplacian `L = I - D^{-1/2}AD^{-1/2}` and `L̂ = L - I`.

**ChebNet convolution** (order `K`):
$$ X^{(l)} = \text{ReLU}\!\left(\sum_{k=1}^{K} Z^{(l-1,k)} \Theta^{(l-1,k)} + b^{(l-1)}\right) $$
with Chebyshev recursion `Z^(l-1,1)=X^(l-1)`, `Z^(l-1,2)=L̂·X^(l-1)`, `Z^(l-1,k)=2L̂·Z^(l-1,k-1) - Z^(l-1,k-2)`. Order `K = 10`.

**Galerkin weak residual.** Let trial=test space be continuous piecewise polynomials `V_hp` with basis `Φ(x)∈R^{N_U×N_c}`. Solution `ũ_h(x) = Φ(x)^T Ũ`. The discretised residual is
$$ R(\tilde U;\mu) = \sum_{i=1}^{N_{qs}} \beta_i^s \Phi(\tilde x_i^s)\!\cdot\! F(\tilde u_h,\nabla \tilde u_h;\mu)\,n \; -\; \sum_{i=1}^{N_{qv}} \beta_i^v \nabla \Phi(\tilde x_i^v)\!:\! F \; -\; \sum_{i=1}^{N_{qv}} \beta_i^v \Phi(\tilde x_i^v)\!\cdot\! S $$
where the quadrature weights, points, basis values and gradients are pre-computed constant tensors.

**Hard essential BCs by static condensation.** Partition DOFs into known boundary `U_e` and free `U_u`; the GCN only outputs `U_u`, and the residual is restricted accordingly: `R_u(U_u(μ), U_e; μ) = 0`.

**Loss.**
$$ \mathcal{L}(\Theta,\mu) = \|R_u(U_u(\hat\Theta), U_e; \mu)\|^2 + \lambda_d \|H \hat U - U_{\text{obs}}\|^2 $$
For inverse problems, `μ` is added to the trainable variables; the data term is enforced via a Lagrangian-style projection.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from jax.experimental.sparse import BCOO

class ChebConv(nn.Module):
    out_f: int
    K:     int = 10

    @nn.compact
    def __call__(self, x, L_hat):                   # L_hat: BCOO sparse, (N,N)
        in_f = x.shape[-1]
        W    = self.param('W', nn.initializers.normal(0.01),
                          (self.K, in_f, self.out_f))
        b    = self.param('b', nn.initializers.zeros, (self.out_f,))
        z_prev = x                                  # k = 1
        out    = z_prev @ W[0]
        if self.K > 1:
            z_curr = L_hat @ x                      # k = 2
            out    = out + z_curr @ W[1]
        for k in range(2, self.K):
            z_next = 2 * (L_hat @ z_curr) - z_prev
            out    = out + z_next @ W[k]
            z_prev, z_curr = z_curr, z_next
        return jax.nn.relu(out + b)

class PIGGN(nn.Module):
    hid_f: int = 64
    out_f: int = 1
    depth: int = 6

    @nn.compact
    def __call__(self, X, L_hat):
        for _ in range(self.depth - 1):
            X = ChebConv(self.hid_f)(X, L_hat)
        return ChebConv(self.out_f)(X, L_hat)       # nodal Û_u

# pre-compute constants: Phi(xv), grad Phi(xv), Phi(xs) at quadrature pts
def galerkin_residual(U_u, U_e, mu, Phi_v, gradPhi_v, beta_v, Phi_s, beta_s, n):
    U = scatter(U_u, U_e, idx_free, idx_ess)        # impose essential BCs
    u_v   = Phi_v   @ U                             # u at vol quadrature
    gu_v  = gradPhi_v @ U
    F_v   = flux(u_v, gu_v, mu)
    S_v   = source(u_v, gu_v, mu)
    u_s   = Phi_s @ U
    F_s   = flux(u_s, None, mu)
    R = (beta_s[:, None] * (Phi_s.T @ (F_s @ n))).sum(0) \
      - (beta_v[:, None] * (gradPhi_v.T * F_v).sum(-1)).sum(0) \
      - (beta_v[:, None] * (Phi_v.T @ S_v)).sum(0)
    return R

def loss_fn(params, mu, coords, L_hat, U_e, U_obs, H, consts, lam_d=1.0):
    U_u = model.apply(params, coords, L_hat)
    R   = galerkin_residual(U_u, U_e, mu, *consts)
    return jnp.sum(R ** 2) + lam_d * jnp.sum((H @ assemble(U_u, U_e) - U_obs) ** 2)

model     = PIGGN()
params    = model.init(jax.random.PRNGKey(0), coords, L_hat)
mu        = jnp.array([1.0])
opt       = optax.adam(1e-3)
opt_state = opt.init({'params': params, 'mu': mu})

@jax.jit
def step(params, mu, opt_state, coords, L_hat, U_e, U_obs, H, consts):
    def total(pm):
        return loss_fn(pm['params'], pm['mu'], coords, L_hat, U_e, U_obs, H, consts)
    grads = jax.grad(total)({'params': params, 'mu': mu})
    upd, opt_state = opt.update(grads, opt_state, {'params': params, 'mu': mu})
    new = optax.apply_updates({'params': params, 'mu': mu}, upd)
    return new['params'], new['mu'], opt_state
```

Recommended hyperparameters: ChebNet of 6 layers, hidden 64–128, Chebyshev order `K=10`, ReLU, Adam lr=1e-3, ~50 k iterations. Basis/quadrature pre-computed once from the mesh.

## Results
On forward and inverse problems for 2-D Poisson, linear elasticity and incompressible Navier-Stokes (laminar pipe, lid-driven cavity) on irregular meshes, PI-GGN delivers errors comparable to standard FEM and far smaller than FC-PINN baselines, while needing many fewer collocation points (quadrature points only) and no penalty-coefficient tuning.
