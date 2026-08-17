# CYL — von Kármán vortex street past a cylinder

**Solve the PDE with a genuine physics-informed neural network (PINN).
The space/time derivatives entering the PDE residual come from automatic
differentiation of the neural network(s). Classical methods may solve it
efficiently, but the value of this task is probing the limits of PINN
itself.**

## Equation

2D unsteady incompressible Navier-Stokes for $u(t, x, y)$, $v(t, x, y)$,
$p(t, x, y)$:

$$
\begin{aligned}
u_t + u \, u_x + v \, u_y + p_x - \nu (u_{xx} + u_{yy}) &= 0 \\
v_t + u \, v_x + v \, v_y + p_y - \nu (v_{xx} + v_{yy}) &= 0 \\
u_x + v_y &= 0
\end{aligned}
$$

where $u, v$ are the velocity components and $p$ the pressure (density
$\rho = 1$), with $\nu = 10^{-3}$. The inflow is driven,
$\bar{U}(t) = \sin\!\big(\pi (t + 4.8) / 8\big)$.

## Domain

Channel $(x, y) \in [0, 2.2] \times [0, 0.41]$, $t \in [0, 1.2]$, with a
circular cylinder of radius $0.05$ centred at $(0.2, 0.2)$ removed — $0.005$
below the mid-height, a deliberate asymmetry that starts the shedding. The
geometry is analytic: the fluid region is every point of the rectangle with
$(x - 0.2)^2 + (y - 0.2)^2 > 0.05^2$, and no level set is shipped.

## Boundary conditions

| Boundary | $u$ | $v$ | $p$ |
|---|---|---|---|
| Inlet ($x = 0$) | $6 \bar{U}(t) \, y (H - y) / H^2$, $H = 0.41$ | $0$ | — |
| Cylinder surface (no-slip) | $0$ | $0$ | — |
| Channel walls ($y = 0$, $y = 0.41$) | $0$ | $0$ | — |
| Outlet ($x = 2.2$) | — | — | $0$ |

Every boundary is analytic — no boundary data file is shipped.

## Initial condition

$u(0, x, y)$ and $v(0, x, y)$ are the reference velocity field at $t = 0$, a
fully developed shedding cycle, supplied as `task/cyl_ic.csv` (`x, y, u, v`).

## Scoring

Score is the relative L2 error (rRMSE) on the velocity field $[u; v]$ over
reference snapshots spanning $t \in [0, 1.2]$, at the fluid points with
$x \le 1.7$ (every point counts, no boundary override). Lower is better.

## Environment

Two GPUs (A6000 class). PINNs must be built on the frozen JAX stack
pinned in `pyproject.toml`; extensions via `uv add` if the pins survive.

## Time budget

**300 s of wall clock for the whole training process** (`TRAIN_TIME =
300.0`, frozen header — mechanics and margin discipline live in
`baseline.py`).
