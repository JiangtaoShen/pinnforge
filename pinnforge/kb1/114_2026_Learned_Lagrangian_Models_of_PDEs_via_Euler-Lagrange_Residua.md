---
slot: 114
title: "Learned Lagrangian Models of PDEs via Euler-Lagrange Residual Minimization (ELM)"
authors: [Lyra Zhornyak, Eric Forgoston, M. Ani Hsieh]
year: 2026
venue: arXiv:2605.07157
gitrepo: ""
---

## TL;DR
A *near-symplectic* mesh-free integrator for forward simulation of PDEs whose continuous Lagrangian density `L(q, q_t, q_x, ...)` is learned by a neural network. Instead of fixed time-stepping, advance the state by minimising the squared Euler-Lagrange residual on Hermite-interpolated local space-time patches with damped Newton steps, swept globally by Jacobi iteration. Symplectic in the zero-residual limit; linear scaling in domain size; no architectural constraints on the learned `L`.

## Problem
Lagrangian / Hamiltonian Neural Networks learn conservation structure but couple it to a fixed integrator (typically Euler or RK4) that does *not* preserve symplecticity, so long-horizon forecasts drift in energy. Prior discrete-Lagrangian learning (Sæmundsson et al., Offen et al.) is limited to ODEs or 1-D PDEs with global implicit solves. For learned *continuous* Lagrangian field densities (especially in 2-D PDEs), no symplectic forward integrator existed.

## Method

### Euler-Lagrange residual
For a learned Lagrangian density `L: TQ → R` and path `q(t,x)`:
$$
R[q\mid (t,x)]=\frac{\partial L}{\partial q}-\frac{d}{dt}\frac{\partial L}{\partial q_t}-\frac{d}{dx}\frac{\partial L}{\partial q_x}.
$$
`R=0` is the EL equation. The total derivatives are evaluated by automatic differentiation through the network `L_θ`.

### Patch Hermite interpolation
On a patch with `m` nodes carrying `(q, q_t, q_x, q_{tx})` (4m DoFs), fit `\hat q(t,x)=\sum_{i=1}^{4m}\theta_i f_i(t,x)` by solving the linear system on collocated values and derivatives. Evaluate `R[\hat q]` at `n_ω≥4m` quadrature points; the patch error is
$$
J(Q,X)\;\approx\;\sum_{i=1}^{n_\omega}\omega_i\,R^2[\hat q\mid (t_i,x_i)].
$$
Oversampling (`n_ω>4m`) trades exact symplecticity for stability against imperfect learned `L`.

### Damped Newton on patches + Jacobi global sweep
Unknowns are the next-time node values `γ = {q(t_{i+1}, x_j)}_j` (with `q_t, q_x, q_{tx}` either provided or estimated). Damped Newton step:
$$
\gamma^{(k+1)}=\gamma^{(k)}-\lambda\big(\nabla_\gamma^2 J\big)^{-1}\nabla_\gamma J,\quad \lambda\in(0,1].
$$
Patches overlap; `n_r` Newton iterations are interleaved with Jacobi sweeps across patches—no global solve, so total cost scales `O(N_patches · m³)` per sweep, linear in domain size. Boundary conditions enter as constraints on `γ`.

### Symplecticity & training
When `R=0` and the patch quadrature is exact, ELM is a Galerkin variational integrator and is symplectic. The learned Lagrangian itself can be trained by any method (LNN, supervised on energy / trajectories, PINN-style); only twice-differentiability is required.

```python
import jax, jax.numpy as jnp

def euler_lagrange_residual(L_params, L_apply, q_fn, t, x):
    """Compute R = dL/dq - d/dt(dL/dq_t) - d/dx(dL/dq_x) per point."""
    def q_t_fn(t, x): return jax.grad(lambda tt: q_fn(tt, x))(t)
    def q_x_fn(t, x): return jax.grad(lambda xx: q_fn(t, xx))(x)
    def Lval(t, x):
        q  = q_fn(t, x)
        qt = q_t_fn(t, x)
        qx = q_x_fn(t, x)
        return L_apply(L_params, q, qt, qx)
    dLdq  = jax.grad(lambda q,  qt, qx: L_apply(L_params, q,  qt, qx), 0)
    dLdqt = jax.grad(lambda q,  qt, qx: L_apply(L_params, q,  qt, qx), 1)
    dLdqx = jax.grad(lambda q,  qt, qx: L_apply(L_params, q,  qt, qx), 2)
    def per_pt(t, x):
        q  = q_fn(t, x); qt = q_t_fn(t, x); qx = q_x_fn(t, x)
        ddt = jax.grad(lambda tt: dLdqt(q_fn(tt, x), q_t_fn(tt, x), q_x_fn(tt, x)))(t)
        ddx = jax.grad(lambda xx: dLdqx(q_fn(t, xx), q_t_fn(t, xx), q_x_fn(t, xx)))(x)
        return dLdq(q, qt, qx) - ddt - ddx
    return jax.vmap(per_pt)(t, x)

def patch_error(L_params, L_apply, Q_nodes, X_nodes, quad_t, quad_x, weights, basis_fns):
    theta = jnp.linalg.solve(build_hermite_matrix(basis_fns, X_nodes), Q_nodes)
    def qhat(t, x):
        return sum(theta[i] * basis_fns[i](t, x) for i in range(theta.shape[0]))
    R = euler_lagrange_residual(L_params, L_apply, qhat, quad_t, quad_x)
    return jnp.sum(weights * R**2)

def elm_step(L_params, L_apply, state, dt, n_newton=5, damp=0.7):
    gamma = warm_start(state, dt)
    def J_fn(g):
        return sum(patch_error(L_params, L_apply, p.Q(g), p.X(),
                               p.quad_t, p.quad_x, p.w, basis_fns)
                   for p in state.patches)
    for _ in range(n_newton):
        g_grad = jax.grad(J_fn)(gamma)
        H      = jax.hessian(J_fn)(gamma)                # patch-local
        gamma  = gamma - damp * jnp.linalg.solve(H, g_grad)
    return gamma
```

Hyper-parameters: 4-node Hermite patch (`m=4`, 16 DoFs), `n_ω≈25–49` quadrature points, `n_r=3–5` Newton iters per sweep, damp `λ=0.7`, MLP `L_θ` width 64 with `tanh` (must be C²).

## Results
On a chaotic double-pendulum (ODE), ELM energy drift matches state-of-the-art symplectic integrators; on the 1-D wave equation, a learned Lagrangian density outperforms Lagrangian Neural Networks in both accuracy and compute time; on the 2-D wave equation, ELM provides the first long-horizon symplectic forward simulation with a learned continuous Lagrangian, correctly producing reflection / transmission / interference under spatially varying dynamics and unseen boundary conditions without retraining.
