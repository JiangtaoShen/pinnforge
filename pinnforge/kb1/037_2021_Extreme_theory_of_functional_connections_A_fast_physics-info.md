---
slot: 037
title: "Extreme theory of functional connections: A fast physics-informed neural network method for solving ordinary and partial differential equations"
authors: [E. Schiassi, R. Furfaro, Carl Leake, Mario De Florio, Hunter Johnston, D. Mortari]
year: 2021
venue: Neurocomputing 457
gitrepo: ""
doi: 10.1016/j.neucom.2021.06.015
---

## TL;DR
X-TFC fuses the Theory of Functional Connections (TFC) — a constrained-expression that analytically enforces IC/BC — with a single-hidden-layer Extreme Learning Machine (ELM) as the free function. Training reduces to (iterative) linear least-squares on the output weights only, giving PINN-level accuracy in milliseconds without IC/BC penalty terms.

## Problem
Standard PINNs minimize a composite loss that includes IC/BC penalties; the resulting gradient-descent training is slow and the constraints are only approximately satisfied. Classical TFC with orthogonal polynomials suffers from the curse of dimensionality in PDEs.

## Method
The latent solution is written as a constrained expression
$$ f(x,\Theta) = A(x) + B(x, g(x)) $$
where `A(x)` analytically satisfies all IC/BC and `B` projects the free function `g(x)` onto the space of functions that vanish at the constraints. The free function is an ELM:
$$ g(x) = \sum_{j=1}^{L} \beta_j\,\sigma(w_j^\top x + b_j) = \sigma^\top \beta $$
with `w_j, b_j` drawn from `U(-10,10)` and frozen, and only `β` trainable. Because `A` already encodes IC/BC, the loss contains the PDE residual alone:
$$ \mathcal{L} = c f_t + N[f;\lambda] - U $$
For linear PDEs collocation yields `Aβ = b` solved by `β = (A^T A)^{-1} A^T b`. For nonlinear PDEs, iterative LS:
$$ \beta_{k+1} = \beta_k - (J^\top J)^{-1} J^\top \mathcal{L}(\beta_k) $$
where `J = ∂L/∂β`.

Example (1-D heat, BCs `f(t,0)=f(t,1)=0`, IC `f(0,x)=sin(πx)`):
$$ f(t,x) = g(t,x) + (x-1)g(t,0) - x g(t,1) - g(0,x) + (1-x)g(0,0) + x g(0,1) + \sin(\pi x) $$

JAX (jax.numpy + jax.jacrev):
```python
import jax, jax.numpy as jnp

L_hidden, n_in = 50, 2                          # hidden neurons; (t,x)
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
W = jax.random.uniform(k1, (L_hidden, n_in), minval=-10.0, maxval=10.0)  # frozen
b = jax.random.uniform(k2, (L_hidden,),      minval=-10.0, maxval=10.0)  # frozen

def sigma(z): return jax.nn.sigmoid(z)

def g(t, x):
    z = t * W[:, 0] + x * W[:, 1] + b           # broadcast: (N, L)
    return sigma(z)

def f_CE(t, x, beta):
    s    = g(t, x) @ beta
    s_t0 = g(t, jnp.zeros_like(x)) @ beta
    s_t1 = g(t, jnp.ones_like(x))  @ beta
    s_0x = g(jnp.zeros_like(t), x) @ beta
    s_00 = g(jnp.zeros_like(t), jnp.zeros_like(x)) @ beta
    s_01 = g(jnp.zeros_like(t), jnp.ones_like(x))  @ beta
    return (s + (x-1)*s_t0 - x*s_t1 - s_0x
              + (1-x)*s_00 + x*s_01 + jnp.sin(jnp.pi*x))

# residual operator for the heat equation u_xx - u_t = 0
def residual(beta, t, x):
    u    = lambda tt, xx: f_CE(tt, xx, beta)
    u_t  = jax.grad(lambda tt, xx: u(tt, xx).sum(), 0)
    u_x  = jax.grad(lambda tt, xx: u(tt, xx).sum(), 1)
    u_xx = jax.grad(lambda tt, xx: u_x(tt, xx).sum(), 1)
    return u_xx(t, x) - u_t(t, x)

beta = jnp.zeros(L_hidden)
t = jnp.repeat(jnp.linspace(0, 1, 50), 50)[:, None]
x = jnp.tile  (jnp.linspace(0, 1, 50), 50)[:, None]

for k in range(20):                              # iterative LS
    res = residual(beta, t, x).squeeze()
    J   = jax.jacrev(lambda b: residual(b, t, x).squeeze())(beta)   # (N, L)
    dbeta = jnp.linalg.lstsq(J, -res)[0]
    beta  = beta + dbeta
```
Hyperparameters: `N = 50` collocation points, `L = 50` neurons, logistic activation, weights `U(-10,10)`. Run 10^3 Monte-Carlo seeds for robustness.

## Results
On linear/nonlinear ODEs and bivariate PDEs (heat, Burgers), X-TFC matches classical TFC (machine precision) and outperforms Deep-TFC on smooth PDEs by orders of magnitude in error and 10–100× in wall-clock time. Deep-TFC is preferred only for highly non-smooth problems (e.g. Navier–Stokes).
