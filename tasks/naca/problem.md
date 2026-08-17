# NACA — flow over a NACA0012 airfoil

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
dimensionless), with $\nu = 1/\text{Re}$, $\text{Re} = 1000$
($\nu = 1.0 \times 10^{-3}$).

## Domain

Farfield box $(x, y) \in [-3, 5] \times [-2, 2]$ with a NACA0012 airfoil
(chord $[0, 1]$, 12% thickness) at 7° angle of attack (rotated $-7°$ about
$(0.5, 0)$). The fluid region is the box minus the airfoil, given by the
signed-distance level set `phi` in `task/collocation.csv` (`x, y, phi`;
`phi <= 0` = fluid, `phi > 0` = airfoil interior).

## Boundary conditions

| Boundary | $u$ | $v$ |
|---|---|---|
| Airfoil surface (no-slip) | $0$ | $0$ |
| Farfield outer box (freestream) | $1$ | $0$ |

Geometry sources: the airfoil no-slip points are analytic —
`_naca_airfoil_points()` in `task/baseline.py`; the fluid mask is the
`phi` column of `task/collocation.csv`.

## Initial condition

None (steady-state problem; no time variable).

## Scoring

Score is the relative L2 error (rRMSE) on the velocity magnitude
$V = \sqrt{u^2 + v^2}$, over the fluid points (`phi <= 0`) in the
near-airfoil window $x \in [-0.2, 2.0]$, $y \in [-0.25, 0.25]$. Lower is
better.

## Environment

Two GPUs (A6000 class). PINNs must be built on the frozen JAX stack
pinned in `pyproject.toml`; extensions via `uv add` if the pins survive.

## Time budget

**180 s of wall clock for the whole training process** (`TRAIN_TIME =
180.0`, frozen header — mechanics and margin discipline live in
`baseline.py`).
