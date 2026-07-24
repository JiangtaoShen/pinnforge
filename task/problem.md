# LDC — Lid-Driven Cavity

**Solve the PDE with a genuine physics-informed neural network (PINN).
The space/time derivatives entering the PDE residual come from automatic
differentiation of the neural network(s). Classical methods may solve it
efficiently, but the value of this task is probing the limits of PINN
itself.**

## Equation

2D steady incompressible Navier-Stokes for $u(x, y)$, $v(x, y)$, $p(x, y)$:

$$
\begin{aligned}
u \, u_x + v \, u_y + p_x - \nu (u_{xx} + u_{yy}) &= 0 \\
u \, v_x + v \, v_y + p_y - \nu (v_{xx} + v_{yy}) &= 0 \\
u_x + v_y &= 0
\end{aligned}
$$

where $u, v$ are the velocity components and $p$ the pressure (all
dimensionless), with $\nu = 1/\text{Re}$, $\text{Re} = 3200$
($\nu \approx 3.125 \times 10^{-4}$).

## Domain

Unit square cavity: $(x, y) \in [0, 1] \times [0, 1]$.

## Boundary conditions

| Edge | $u$ | $v$ |
|---|---|---|
| Top ($y = 1$) | $1$ | $0$ |
| Bottom ($y = 0$) | $0$ | $0$ |
| Left ($x = 0$) | $0$ | $0$ |
| Right ($x = 1$) | $0$ | $0$ |

## Initial condition

None (steady-state problem; no time variable).

## Scoring

Score is the relative L2 error (rRMSE) on the velocity field $[u; v]$.
Lower is better. Boundary predictions are replaced by the known BC values
before scoring.

## Environment

Two GPUs (A6000 class). PINNs must be built on the frozen JAX stack
pinned in `pyproject.toml`; extensions via `uv add` if the pins survive.

## Time budget

**150 s of wall clock for the whole training process** (`TRAIN_TIME =
150.0`, frozen header — mechanics and margin discipline live in
`baseline.py`).
