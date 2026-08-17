---
slot: 10
title: "Variational Physics-Informed Neural Networks For Solving Partial Differential Equations"
authors: [Ehsan Kharazmi, Zhongqiang Zhang, George Em Karniadakis]
year: 2019
venue: "arXiv:1912.00873"
gitrepo: ""
---

## TL;DR
A Petrov-Galerkin PINN: the *trial* function is a deep neural net, but the *test* functions are a fixed linear basis (Legendre polynomials, sines, or compactly-supported polynomials on subdomains). The loss is the squared variational residual `<L u_NN - f, v_k>` for each test function, after integration by parts to lower the derivative order on `u_NN`. Cheaper higher derivatives + spectral-grade accuracy.

## Problem
Strong-form PINN evaluates the PDE residual pointwise at random collocation points (equivalent to using Dirac-delta test functions), and must compute high-order autograd derivatives of a deep network at every point. This is expensive and noisy near singularities.

## Method
For a steady PDE `L_q u = f` on `Omega`, BC `u = h` on `dOmega`, set `u_NN(x; w, b)` as trial. Choose `K` test functions `{v_k}_{k=1..K}` (compactly supported on `Omega`, e.g. shifted Legendre, sine basis, or local FEM-style hats on a partition of `Omega`).

Variational residual (after one or more integration-by-parts):
$$
R_v[u_{NN}; v_k] = \int_\Omega \bigl(L_q u_{NN}\bigr)\,v_k\,d\Omega - \int_\Omega f\,v_k\,d\Omega
$$
For e.g. `L = -Delta`, by parts:
$$
R_v = \int_\Omega \nabla u_{NN}\cdot\nabla v_k\,d\Omega - \int_{\partial\Omega} (\nabla u_{NN}\cdot n)\,v_k\,d\sigma - \int_\Omega f v_k\,d\Omega
$$
With `v_k` compactly supported, the boundary term drops. Loss:
$$
\mathcal{L}_v = \sum_{k=1}^{K} |R_v[u_{NN}; v_k]|^2 \;+\; \tau\,\frac{1}{N_u}\sum_i |u_{NN}(x_i^u) - h(x_i^u)|^2
$$
The integrals are approximated by a quadrature rule (Gauss-Legendre with `n_q` points). When `Omega` is split into `N_el` subdomains (h-p VPINN), per element use a local set of test functions of degree up to `p_el`, allowing local adaptivity.

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    h: int = 20
    depth: int = 3
    act: callable = jnp.sin
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = self.act(nn.Dense(self.h)(x))
        return nn.Dense(1)(x)

def legendre_basis(x, K):
    # x shape (N, 1) in [-1, 1]; returns (N, K) matrix of P_0..P_{K-1}
    P0 = jnp.ones_like(x); P1 = x
    out = [P0, P1]
    Pn_1, Pn = P0, P1
    for n in range(1, K - 1):
        Pn_p1 = ((2 * n + 1) * x * Pn - n * Pn_1) / (n + 1)
        out.append(Pn_p1); Pn_1, Pn = Pn, Pn_p1
    return jnp.concatenate(out, axis=1)[:, :K]

net = MLP()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 1)))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

xq, wq = gauss_legendre(64)               # (N,1), (N,)
V     = legendre_basis(xq, K=20)          # (N, K)
dV_dx = legendre_basis_derivative(xq, K=20)
tau   = 10.0
x_bc, h_bc = boundary_data()

def u_apply(p, x):  return net.apply(p, x)[:, 0]

def loss_fn(params):
    def u_single(p, xi):  return u_apply(p, xi[None])[0]
    u_x = jax.vmap(lambda xi: jax.grad(u_single, argnums=1)(params, xi))(xq[:, 0])[:, None]
    # weak form for -u'' = f  --> integrate u' * v'
    Rv = jnp.sum(u_x * dV_dx * wq[:, None], axis=0)              # (K,)
    f_int = jnp.sum(f(xq) * V * wq[:, None], axis=0)             # (K,)
    L_R  = jnp.sum((Rv - f_int) ** 2)
    L_bc = jnp.mean((u_apply(params, x_bc) - h_bc) ** 2)
    return L_R + tau * L_bc

@jax.jit
def train_step(params, opt_state):
    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

Variants:
- Shallow + sine activation + sine test functions: integrand becomes analytic, residuals are closed form (no quadrature).
- h-p VPINN: partition `Omega = U_e Omega_e`; build local `K_e` test functions per element; sum residuals across elements; gives p-refinement near features.

Recommended: depth 2-4, width 20-50; K = 10-40 test functions per dimension; quadrature order `n_q >= 2 p + 1`; `tau ~ 1-100` for BC penalty; Adam `lr=1e-3` then L-BFGS.

## Results
On 1-D Poisson with smooth and rough solutions, steady Burgers, and 2-D Poisson, VPINN reaches relative L2 of `1e-4` to `1e-6` with 5-10x fewer derivative evaluations than equivalent-accuracy PINN. h-p VPINN with local test functions handles solutions with localised steep gradients where global VPINN saturates.
