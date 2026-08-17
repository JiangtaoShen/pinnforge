---
slot: 74
title: "Challenges in Training PINNs: A Loss Landscape Perspective"
authors: [Pratik Rathore, Weimu Lei, Zachary Frangella, Lu Lu, Madeleine Udell]
year: 2024
venue: "ICML 2024 (arXiv:2402.01868)"
gitrepo: "https://github.com/pratikrathore8/opt_for_pinns"
---

## TL;DR
The PINN loss Hessian is ill-conditioned (top eigenvalue 1e4-1e5) because of the differential operator in the residual term. The authors show that **Adam followed by L-BFGS** beats either alone, and that L-BFGS often terminates early at a non-critical point; appending a damped Nystrom-preconditioned Newton-CG (**NNCG**) recovers another 10x reduction in loss.

## Problem
For PINN loss `L(w) = (1/2n_r) sum ||D[u(x_i;w)]||^2 + (1/2n_b) sum ||B[u(x_j;w)]||^2`, the differential operator `D` makes `H_L` ill-conditioned with fast spectral decay. First-order optimisers (Adam) converge slowly, while quasi-Newton (L-BFGS) gets stuck at saddles and at strong-Wolfe step-size = 0, leaving the gradient norm at 1e-2 to 1e-3.

## Method

### Pipeline: Adam -> L-BFGS -> NNCG
Adam first escapes saddles (first-order methods provably avoid strict saddles); then L-BFGS preconditions the local landscape (reduces top eigenvalue >=1e3); then NNCG performs damped Newton steps via Nystrom-preconditioned CG on the exact Hessian-vector product, with Armijo line search to avoid L-BFGS's strong-Wolfe deadlock.

### NysNewton-CG (NNCG)
At iterate `w`, form a Nystrom sketch of `H_L(w)` using `r` random Hessian-vector products, build the preconditioner `P = U(Lambda + mu I)^-1 U^T + mu^-1 (I - UU^T)`, run preconditioned CG to solve `(H_L + rho I) d = -grad L`, then Armijo backtracking.

Key formulas:
$$ H v = \nabla_w (\nabla_w L(w)^\top v) \quad \text{(HVP, } O((n_r+n_b)p) \text{)} $$
$$ \Omega = \texttt{randn}(p, r),\ Y = H\Omega,\ [Q,R] = QR(Y),\ \tilde C = Q^\top H Q,\ U\Lambda U^\top = \texttt{eigh}(\tilde C) $$

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from jax.flatten_util import ravel_pytree

def hvp(loss_fn, params, v):                        # HVP via grad of grad-vector product
    return jax.grad(lambda p: jnp.vdot(jax.grad(loss_fn)(p), v))(params)

def nystrom_sketch(hvp_fn, params, key, r=20):
    flat, unravel = ravel_pytree(params)
    p = flat.size
    Omega = jax.random.normal(key, (p, r))
    Y = jnp.stack([ravel_pytree(hvp_fn(unravel(Omega[:, j])))[0] for j in range(r)], axis=1)
    Q, _  = jnp.linalg.qr(Y)
    C = jnp.stack([ravel_pytree(hvp_fn(unravel(Q[:, j])))[0] for j in range(r)], axis=1)
    C = 0.5 * (C + C.T)
    L, U = jnp.linalg.eigh(Q.T @ C)
    return Q @ U, L                                 # (p,r), (r,)

def nncg_step(loss_fn, params, key, rho=1e-4, r=20, max_cg=20):
    flat_g, unravel = ravel_pytree(jax.grad(loss_fn)(params))
    hvp_fn = lambda v_tree: hvp(loss_fn, params, v_tree)
    U, L = nystrom_sketch(hvp_fn, params, key, r)
    mu = L[-1]
    def P_inv(x):                                   # Nystrom preconditioner
        return U @ ((1.0/(L+rho)) * (U.T @ x)) + (x - U @ (U.T @ x))/(mu+rho)
    def A(x):                                       # (H + rho I) v
        return ravel_pytree(hvp_fn(unravel(x)))[0] + rho * x
    d = pcg(A, -flat_g, M=P_inv, max_iter=max_cg)   # preconditioned CG
    alpha = 1.0                                     # Armijo backtracking
    L0 = loss_fn(params); gd = jnp.vdot(flat_g, d)
    while loss_fn(unravel(flat_g*0 + ravel_pytree(params)[0] + alpha*d)) > L0 - 1e-4*alpha*gd:
        alpha *= 0.5
    return unravel(ravel_pytree(params)[0] + alpha*d)

# Training pipeline
opt_a = optax.adam(1e-3); state_a = opt_a.init(params)
for _ in range(1000):
    grads = jax.grad(loss_fn)(params)
    upd, state_a = opt_a.update(grads, state_a); params = optax.apply_updates(params, upd)
opt_l = optax.lbfgs(); state_l = opt_l.init(params)
for _ in range(30000):
    grads = jax.grad(loss_fn)(params)
    upd, state_l = opt_l.update(grads, state_l, params,
        value=loss_fn(params), grad=grads, value_fn=loss_fn)
    params = optax.apply_updates(params, upd)
key = jax.random.PRNGKey(0)
for _ in range(N_nncg):
    key, sk = jax.random.split(key); params = nncg_step(loss_fn, params, sk)
```

Hyperparameters: tanh MLP, 3 hidden layers x 50-400; Adam lr in `{1e-5,...,1e-1}`; switch to L-BFGS after 1k/11k/31k Adam steps; total 41k; NNCG rank `r=60`, damping `rho=1e-4`, preconditioner update every 20 steps.

## Results
On convection (beta=40), reaction (rho=5) and wave (beta=5) PDEs, Adam+L-BFGS yields the lowest L2RE for every width (e.g. 14.2x smaller than Adam on convection, 6.07x smaller than L-BFGS on wave). Appending NNCG reduces loss by >10x further and gradient norm by 1-2 orders of magnitude on convection and wave.
