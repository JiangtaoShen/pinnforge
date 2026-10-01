# POISSON — Multi-scale Poisson (demo)

**Solve the PDE with a genuine physics-informed neural network (PINN).
The space/time derivatives entering the PDE residual come from automatic
differentiation of the neural network(s). Classical methods may solve it
efficiently, but the value of this task is probing the limits of PINN
itself.**

## Equation

2D Poisson equation for $u(x, y)$:

$$
-\left(u_{xx} + u_{yy}\right) = f(x, y), \qquad
f = 2\pi^2 \sin(\pi x)\sin(\pi y) + 12.8\,\pi^2 \sin(8\pi x)\sin(8\pi y)
$$

The source carries two scales: the smooth mode $2\pi^2 \sin(\pi x)\sin(\pi y)$
and a mode eight times finer in each direction, $12.8\,\pi^2 \sin(8\pi x)\sin(8\pi y)$,
whose amplitude is 6.4 times the smooth mode's.

## Domain

Unit square: $(x, y) \in [0, 1] \times [0, 1]$.

## Boundary conditions

Homogeneous Dirichlet on all four edges: $u = 0$.

## Initial condition

None (steady-state problem; no time variable).

## Scoring

Score is the relative L2 error (rRMSE) on $u$ over a uniform
$101 \times 101$ grid covering the whole square, edges included. Lower is
better.

## Environment

The GPU pool and the per-block budget are stated in your charter
(`block.md`). PINNs must be built on the frozen JAX stack pinned in
`pyproject.toml`; extensions via `uv add` if the pins survive.

## Time budget

**60 s of wall clock for the whole training process** (`TRAIN_TIME =
60.0`, frozen header — mechanics and margin discipline live in
`baseline.py`).
