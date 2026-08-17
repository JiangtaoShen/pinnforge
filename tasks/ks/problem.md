# KS — Kuramoto–Sivashinsky

**Solve the PDE with a genuine physics-informed neural network (PINN).
The space/time derivatives entering the PDE residual come from automatic
differentiation of the neural network(s). Classical methods may solve it
efficiently, but the value of this task is probing the limits of PINN
itself.**

## Equation

Scaled Kuramoto–Sivashinsky equation for $u(t, x)$:

$$
u_t + \nu_1\, u\, u_x + \nu_2\, u_{xx} + \nu_3\, u_{xxxx} = 0
$$

where $\nu_1 = 6.25$, $\nu_2 = 0.390625$, $\nu_3 \approx 1.526 \times 10^{-3}$
(the scaled coefficients $100/16$, $100/16^2$, $100/16^4$).

## Domain

$(t, x) \in [0, 0.4] \times [0, 2\pi]$.

## Boundary conditions

Periodic in $x$: $u(t, 0) = u(t, 2\pi)$ (and all $x$-derivatives).

## Initial condition

$u(0, x)$ is the reference profile at $t = 0$, supplied as
`task/ks_ic.csv`.

## Scoring

Score is the relative L2 error (rRMSE) on $u$. Lower is better.

## Environment

Two GPUs (A6000 class). PINNs must be built on the frozen JAX stack
pinned in `pyproject.toml`; extensions via `uv add` if the pins survive.

## Time budget

**300 s of wall clock for the whole training process** (`TRAIN_TIME =
300.0`, frozen header — mechanics and margin discipline live in
`baseline.py`).
