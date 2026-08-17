---
slot: 059
title: "Physics-informed neural networks and functional interpolation for stiff chemical kinetics"
authors: [Mario De Florio, Enrico Schiassi, Roberto Furfaro]
year: 2022
venue: Chaos 32, 063107
gitrepo: ""
doi: 10.1063/5.0086649
---

## TL;DR
**X-TFC** combines (a) the Theory of Functional Connections — write the trial solution as a *constrained expression* that **analytically satisfies the IC** for free — with (b) an Extreme Learning Machine: a single hidden layer with random frozen input weights and only the output weights `beta` trained. With ICs removed from the loss the residual is the only objective; nonlinear least-squares on `beta` solves stiff chemical-kinetics ODE systems (ROBER, POLLU, Belousov-Zhabotinsky) to machine precision in seconds.

## Problem
Stiff ODE systems (eigenvalue spread `>10^6`) defeat plain PINNs: the IC penalty and the residual loss compete and the back-propagated gradients are wildly unbalanced. Existing fixes (Stiff-PINN: quasi-steady-state reduction; neural ODEs) lose accuracy or are slow. The authors want IC-exact, fast, and stiff-robust.

## Method
For a first-order IVP `dy/dt = f(y,t)`, `y(0) = y_0`, the **constrained expression (CE)** is
$$
y_\beta(t) = g_\beta(t) + \big(y_0 - g_\beta(0)\big)
$$
which exactly satisfies `y(0) = y_0` for any `g`. So the residual is the only error term:
$$
R_\beta(t) = \frac{d}{dt} g_\beta(t) - f\!\big(g_\beta(t) + y_0 - g_\beta(0),\, t\big)
$$
`g_beta(t)` is a single-hidden-layer ELM: `g_beta(t) = sum_j beta_j * sigma(w_j x + b_j)` where `w_j, b_j` are sampled once from `U(-1,1)` and frozen; only `beta in R^{L}` (typically `L = 10..200`) is trained. Time is mapped `x = c(t - t0)` with `c = 2/(t_f - t_0)` to put `x in [-1,1]`. Loss on `N` quadrature/collocation points:
$$
\mathcal{L}(\beta) = \sum_{i=1}^N w_i\,|R_\beta(t_i)|^2
$$
Because `g_beta` is linear in `beta`, the loss is quadratic for linear ODEs and is solved by **Gauss-Newton** for nonlinear: `beta_{k+1} = beta_k - (J^T J)^{-1} J^T L(beta_k)` until `||L|| < eps`. For long-time stiff problems the time interval is split into many sub-domains; the final value of one segment becomes `y_0` of the next.

```python
import jax, jax.numpy as jnp
import optax

def init_xtfc(key, L, n_species):
    kW, kb, kbeta = jax.random.split(key, 3)
    W    = jax.random.uniform(kW, (1, L), minval=-1.0, maxval=1.0)   # frozen
    b    = jax.random.uniform(kb, (1, L), minval=-1.0, maxval=1.0)   # frozen
    beta = jnp.zeros((L, n_species))
    return {"W": W, "b": b, "beta": beta}

def sigma  (x, W, b): return jnp.tanh(x @ W + b)                     # [N, L]
def sigma_p(x, W, b): return (1.0 - jnp.tanh(x @ W + b)**2) * W      # [N, L]

def y_pred(params, x, x0, y0):                                       # CE
    g  = sigma(x,  params["W"], params["b"]) @ params["beta"]
    g0 = sigma(x0, params["W"], params["b"]) @ params["beta"]
    return g + (y0 - g0)

def dydx(params, x):
    return sigma_p(x, params["W"], params["b"]) @ params["beta"]     # [N, n]

def rhs(y, t):                                                       # chemical kinetics RHS
    ...                                                              # ROBER, POLLU, ...
    return dy

def loss_segment(params, x, x0, y0, c, t_phys):
    y    = y_pred(params, x, x0, y0)
    dydt = c * dydx(params, x)
    R    = dydt - jax.vmap(rhs)(y, t_phys)
    return jnp.mean(R**2)

def train_segment(key, t0, tf, y0, L=50, N=80, max_iter=50):
    params = init_xtfc(key, L, len(y0))
    c      = 2.0 / (tf - t0)
    x      = jnp.linspace(-1.0, 1.0, N).reshape(-1, 1)
    x0     = jnp.array([[-1.0]])
    t_phys = t0 + (x + 1.0) / c
    y0v    = jnp.asarray(y0).reshape(1, -1)
    # Gauss-Newton via lstsq on the residual Jacobian wrt beta
    def residual_vec(beta_flat):
        p2 = {**params, "beta": beta_flat.reshape(params["beta"].shape)}
        y    = y_pred(p2, x, x0, y0v)
        dydt = c * dydx(p2, x)
        return (dydt - jax.vmap(rhs)(y, t_phys.squeeze())).reshape(-1)
    beta = params["beta"].reshape(-1)
    for _ in range(max_iter):
        J = jax.jacrev(residual_vec)(beta)
        r = residual_vec(beta)
        dbeta, *_ = jnp.linalg.lstsq(J, r, rcond=None)
        beta = beta - dbeta
        if jnp.linalg.norm(r) < 1e-12: break
    params["beta"] = beta.reshape(params["beta"].shape)
    y_end = y_pred(params, jnp.array([[1.0]]), x0, y0v)
    return params, y_end

# Multi-segment time march for long stiff intervals
key = jax.random.PRNGKey(0)
y = jnp.asarray(initial_conditions)
for (t0, tf) in segments:                                            # e.g. log-spaced
    key, sub = jax.random.split(key)
    params, y = train_segment(sub, t0, tf, list(map(float, y)))
```

Activation: `tanh`. Hyper-params: `L = 30..100`, `N = 30..100` Gauss-Legendre points, Gauss-Newton (or L-BFGS via jaxopt), `tol = 1e-12`. A generalization-error bound `eps_G <= C1 (eps_T^2 + C_quad N^{-alpha})^{1/2}` is proved.

## Results
On ROBER, POLLU (20 species, 25 reactions), Akzo Nobel, and Belousov-Zhabotinsky, X-TFC matches MATLAB's `ode23s`/`RADAU` benchmarks at relative errors `1e-8..1e-12`, with training times of 0.1-2 seconds per segment on a laptop — orders of magnitude faster and more accurate than Stiff-PINN or neural ODE baselines.
