---
slot: 94
title: "A QPINN Framework with Quantum Trainable Embeddings for the Lid-Driven Cavity Problem"
authors: [Nahid Binandeh Dehaghani, Ban Q. Tran, Susan Mengel, Rafal Wisniewski, A. Pedro Aguiar]
year: 2026
venue: arXiv:2605.13892 (quant-ph)
gitrepo: ""
---

## TL;DR
Hybrid quantum-classical PINN for steady 2-D lid-driven cavity (Re up to 1e4): two output fields `p(x,y)` and stream function `psi(x,y)` are read out as Pauli-Z expectations of a hardware-efficient variational quantum circuit. The novelty is a *quantum* trainable embedding -- a small QNN that maps `(x_tilde, y_tilde) in [-1,1]^2` to data-encoding angles before the solver VQC.

## Problem
PINNs on Navier-Stokes lid-driven cavity stall due to nonlinear convection and pressure-velocity coupling. Existing QPINNs use fixed (Chebyshev) or classical-FNN embeddings; the question is whether a fully-quantum learned embedding gives parameter-efficient PINNs (~360 params vs 6594 classical).

## Method
Stream-function formulation: outputs (p, psi); velocities `u = psi_y`, `v = -psi_x` make `div u = 0` exact. Loss enforces 2-D momentum residuals plus weak BCs.

Residuals:
$$
R_x = u u_x + v u_y + p_x - \tfrac1{\mathrm{Re}}(u_{xx}+u_{yy}),\;\;
R_y = u v_x + v v_y + p_y - \tfrac1{\mathrm{Re}}(v_{xx}+v_{yy})
$$
$$
\mathcal L = \mathcal L_{PDE} + \lambda_B(\mathcal L_{\text{wall}} + \mathcal L_{\text{lid}} + p(0,0)^2)
$$

Quantum pipeline. Normalize coords to `[-1,1]`. (1) Embedding QNN `U_embed(x_tilde, y_tilde; theta_Q)` produces angles `alpha_i = pi * <Z_i>`. (2) Encoding `U_enc = prod_i R_y(alpha_i)` on `|0>^Nq`. (3) Solver VQC `U_var(theta_var)` of L layers (Rx/Ry/Rz + nearest-neighbor CNOT). (4) Two independent VQCs, one each for p and psi, read out as sum-of-Pauli-Z.

```python
# requires pennylane + jax
import pennylane as qml
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import jaxopt

Nq, L_var, L_emb = 4, 10, 5
dev = qml.device("default.qubit", wires=Nq)

@qml.qnode(dev, interface="jax", diff_method="parameter-shift")
def embed_qnn(xy, theta_Q):
    for L in range(L_emb):
        for q in range(Nq):
            qml.RY(xy[q % 2], wires=q); qml.RX(theta_Q[L, q, 0], wires=q)
            qml.RY(theta_Q[L, q, 1], wires=q); qml.RZ(theta_Q[L, q, 2], wires=q)
        for q in range(Nq - 1): qml.CNOT(wires=[q, q+1])
    return [qml.expval(qml.PauliZ(q)) for q in range(Nq)]

@qml.qnode(dev, interface="jax", diff_method="parameter-shift")
def solver_vqc(alpha, theta_var):
    for i in range(Nq): qml.RY(alpha[i], wires=i)
    for L in range(L_var):
        for q in range(Nq):
            qml.RX(theta_var[L, q, 0], wires=q)
            qml.RY(theta_var[L, q, 1], wires=q)
            qml.RZ(theta_var[L, q, 2], wires=q)
        for q in range(Nq - 1): qml.CNOT(wires=[q, q+1])
    return qml.expval(sum(qml.PauliZ(q) for q in range(Nq)))

class QPINN(nn.Module):
    @nn.compact
    def __call__(self, x, y):
        theta_Q     = self.param("theta_Q",     nn.initializers.normal(0.1), (L_emb, Nq, 3))
        theta_var_p = self.param("theta_var_p", nn.initializers.normal(0.1), (L_var, Nq, 3))
        theta_var_s = self.param("theta_var_s", nn.initializers.normal(0.1), (L_var, Nq, 3))
        xt, yt = 2*x - 1, 2*y - 1                                       # to [-1,1]
        alpha = jnp.pi * jnp.stack(embed_qnn(jnp.array([xt, yt]), theta_Q))
        p   = solver_vqc(alpha, theta_var_p)
        psi = solver_vqc(alpha, theta_var_s)
        return p, psi

net = QPINN()
params = net.init(jax.random.PRNGKey(0), jnp.array(0.5), jnp.array(0.5))

def momentum_residuals(params, x, y, Re):
    def fields(xv, yv):
        return net.apply(params, xv, yv)
    p_fn   = lambda xv, yv: fields(xv, yv)[0]
    psi_fn = lambda xv, yv: fields(xv, yv)[1]
    u  = jax.vmap(jax.grad(psi_fn, argnums=1))(x, y)
    v  = -jax.vmap(jax.grad(psi_fn, argnums=0))(x, y)
    ux = jax.vmap(jax.grad(lambda xv, yv: jax.grad(psi_fn, argnums=1)(xv, yv), argnums=0))(x, y)
    uy = jax.vmap(jax.grad(lambda xv, yv: jax.grad(psi_fn, argnums=1)(xv, yv), argnums=1))(x, y)
    vx = jax.vmap(jax.grad(lambda xv, yv: -jax.grad(psi_fn, argnums=0)(xv, yv), argnums=0))(x, y)
    vy = jax.vmap(jax.grad(lambda xv, yv: -jax.grad(psi_fn, argnums=0)(xv, yv), argnums=1))(x, y)
    px = jax.vmap(jax.grad(p_fn, argnums=0))(x, y)
    py = jax.vmap(jax.grad(p_fn, argnums=1))(x, y)
    uxx = jax.vmap(jax.grad(lambda xv, yv: jax.grad(lambda a, b: jax.grad(psi_fn, argnums=1)(a, b), 0)(xv, yv), 0))(x, y)
    uyy = jax.vmap(jax.grad(lambda xv, yv: jax.grad(lambda a, b: jax.grad(psi_fn, argnums=1)(a, b), 1)(xv, yv), 1))(x, y)
    vxx = jax.vmap(jax.grad(lambda xv, yv: jax.grad(lambda a, b: -jax.grad(psi_fn, argnums=0)(a, b), 0)(xv, yv), 0))(x, y)
    vyy = jax.vmap(jax.grad(lambda xv, yv: jax.grad(lambda a, b: -jax.grad(psi_fn, argnums=0)(a, b), 1)(xv, yv), 1))(x, y)
    Rx = u*ux + v*uy + px - (uxx + uyy)/Re
    Ry = u*vx + v*vy + py - (vxx + vyy)/Re
    return Rx, Ry

def total_loss(params, X, Y, lam_B=10.0, Re=100.0):
    Rx, Ry = momentum_residuals(params, X, Y, Re)
    p00, _ = net.apply(params, jnp.array(0.0), jnp.array(0.0))
    return jnp.mean(Rx**2 + Ry**2) + lam_B*(wall_loss(params) + lid_loss(params) + p00**2)

solver = jaxopt.LBFGS(fun=total_loss, maxiter=100, linesearch="zoom")
state  = solver.init_state(params, X_int, Y_int)
for _ in range(100):
    params, state = solver.update(params, state, X_int, Y_int)
```

Hyperparameters: 50x50 collocation grid, `Nq` in {2,4,6}, `L_var = 10`, `L_emb = 5`, L-BFGS for 100 epochs, `lambda_B ~ 10`. Total trainable params 360 (vs PINN baseline 4 layers x 32 = 6594).

## Results
At Re=100 with 4 qubits / 10 layers: training loss 1.71 vs classical PINN 2.21 vs FNN-TE-QPINN 1.99. Velocity inference MSE 6.64e-4, L2-rel 9.71e-2; pressure L2-rel 5.51e-1 (harder). At Re>=3000 QNN-TE-QPINN gives the lowest training loss (~1.10) among the three. Parameter count 360 vs 6594 classical PINN.
