---
slot: 027
title: "hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition"
authors: [Ehsan Kharazmi, Zhongqiang Zhang, George Em Karniadakis]
year: 2020
venue: "CMAME (arXiv:2003.05385)"
gitrepo: ""
---

## TL;DR
A single global DNN is the trial space, but the residual is projected onto LOCAL piecewise polynomial test functions on a non-overlapping element partition (sub-domain Petrov-Galerkin). h-refinement = more elements, p-refinement = higher-order Legendre test polynomials. Each integral is evaluated by Gauss quadrature, giving lower derivative orders and smaller losses than PINN's strong-form collocation.

## Problem
Strong-form PINNs differentiate the network twice (or more) via AD — costly and noise-amplifying for high-order or singular PDEs. Global VPINNs cannot localize refinement. Need an hp variant that combines the global flexibility of a DNN trial with the local refinement of FEM-style test bases.

## Method
Take PDE L_q u = f. The NN u_NN(x; W, b) is the global trial. Partition Omega into Nel non-overlapping elements Omega_e; on each element pick K^(e) Legendre test polynomials v_k^(e)(x) = P_{k-1} on a local reference. Elemental variational residual:
$$
R_k^{(e)} = \int_{\Omega_e} \big(L_q u_{NN}(x) - f(x)\big)\,v_k^{(e)}(x)\,d\Omega_e
$$
Use integration by parts to push derivatives onto v_k^(e) (smooth and known analytically), so the NN may only need first derivatives even for second-order PDEs.

Loss:
$$
L^v = \sum_{e=1}^{N_{el}}\tfrac{1}{K^{(e)}}\sum_{k=1}^{K^{(e)}} |R_k^{(e)}|^2 + \tau_b L_b + \tau_0 L_0
$$
Integrals approximated by Gauss-Legendre with Q nodes per element.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from numpy.polynomial.legendre import leggauss

class HPVPINN(nn.Module):
    width: int = 20
    depth: int = 4
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.width)(x))
        return nn.Dense(1)(x)

def legendre_basis(K, xi):
    # xi: (Q,) on [-1,1] -> (Q, K) values, (Q, K) derivatives
    P  = [jnp.ones_like(xi), xi]
    dP = [jnp.zeros_like(xi), jnp.ones_like(xi)]
    for k in range(1, K-1):
        P.append(((2*k+1)*xi*P[-1] - k*P[-2]) / (k+1))
        dP.append(((2*k+1)*(P[-2] + xi*dP[-1]) - k*dP[-2]) / (k+1))
    return jnp.stack(P[:K], -1), jnp.stack(dP[:K], -1)

def variational_loss(params, apply_fn, elems, f, K=10, Q=20):
    qpts_np, qwts_np = leggauss(Q)
    qpts = jnp.asarray(qpts_np); qwts = jnp.asarray(qwts_np)
    V, dV = legendre_basis(K, qpts)         # (Q,K), (Q,K) on ref [-1,1]

    def elem_residual(a, b):
        x_phys = 0.5*(b-a)*qpts + 0.5*(a+b)
        jac = 0.5*(b-a)
        u_fn = lambda x: apply_fn(params, x.reshape(1))[0]
        ux   = jax.vmap(jax.grad(u_fn))(x_phys)            # (Q,)
        # For -u_xx = f, IBP: R_k = int (u_x v_k') - f v_k dx
        # u_x is in physical coords; dV is wrt xi, so dv/dx = dV / jac
        integrand = ux[:, None] * (dV / jac) - f(x_phys)[:, None] * V   # (Q,K)
        return jac * jnp.sum(qwts[:, None] * integrand, axis=0)         # (K,)

    Rks = jax.vmap(lambda ab: elem_residual(ab[0], ab[1]))(elems)       # (Nel,K)
    return jnp.mean(Rks**2)

net = HPVPINN()
params = net.init(jax.random.PRNGKey(0), jnp.zeros(1))
opt = optax.adam(1e-3); state = opt.init(params)

@jax.jit
def step(params, state, elems, x_bc, u_bc, tau_b):
    def total(p):
        Lv = variational_loss(p, net.apply, elems, f_source)
        Lb = jnp.mean((jax.vmap(lambda x: net.apply(p, x)[0])(x_bc) - u_bc)**2)
        return Lv + tau_b * Lb
    g = jax.grad(total)(params)
    upd, state = opt.update(g, state, params)
    return optax.apply_updates(params, upd), state
```

Recommended: 4 hidden layers x 20 tanh neurons, K=10 Legendre tests per element, Q=80 Gauss points per element, tau_b ~ 1, Adam + L-BFGS. h-refinement: split elements where residual is large; p-refinement: increase K locally.

## Results
On smooth analytic targets hp-VPINN converges spectrally in p (~O(K^{-p}) error). On 1D/2D elliptic problems and inverse advection-diffusion, hp-VPINN matches or beats VPINN and PINN at equal cost; domain decomposition stabilizes near-discontinuous solutions.
