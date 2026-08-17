"""Vanilla PINN baseline for the DFG 2D-3 cylinder benchmark (tree root).

Contract (do not change):
  - ``predict_fn(params, X) -> {"u","v","p"}``
  - ``train(rng, eval_callback=None) -> (params, step_count)``
  - Frozen PDE-CONSTANTS header is byte-identical across all descendants.

Time budget (frozen): ``TRAIN_TIME = 300`` s of wall-clock for the WHOLE
process, anchored at ``_T0`` (process start) — imports, JIT compilation,
training and param save all included; ``eval.py`` kills the process at this
wall. ``train()`` must return params with margin to spare (this baseline
stops ``SAVE_MARGIN_S = 5`` s early).

Data discipline: the ONLY field data this baseline (and any descendant)
may read is the initial condition ``task/cyl_ic.csv`` (u, v at t=0).
Never read ``task/ref_data.csv`` — it is the scoring truth, used by
``task/eval.py`` for scoring alone. Every boundary is analytic, so no
boundary data file exists to read either.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 3
OUTPUT_DIM   = 3
INPUT_NAMES  = ("t", "x", "y")
FIELD_NAMES  = ("u", "v", "p")

# --- Domain ---
T_MIN, T_MAX = 0.0, 1.2
X_MIN, X_MAX = 0.0, 2.2
Y_MIN, Y_MAX = 0.0, 0.41
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("inlet", "outlet", "walls", "cylinder")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "channel_with_cylinder"

# --- Geometry (DFG 2D benchmark, Schaefer & Turek) ---
CYL_X, CYL_Y = 0.2, 0.2      # cylinder centre; offset from the channel
CYL_R        = 0.05          # mid-height 0.205 to break the symmetry

# --- Physical parameters (DFG 2D-3, Schaefer & Turek: driven inflow) ---
U_PEAK  = 1.5        # peak centreline inflow; U(s) = U_PEAK*sin(pi*s/T_DRIVE)
T_DRIVE = 8.0        # driving period of the benchmark, in simulation time s
T_START = 4.8        # task t = 0 is benchmark time s = T_START
NU      = 1.0e-3     # kinematic viscosity (rho = 1)
RE      = (2.0 / 3.0) * U_PEAK * (2.0 * CYL_R) / NU   # = 100 at the drive peak
                     # the window itself runs at Re(t) = 95 -> 71

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 300.0     # total wall-clock seconds for the WHOLE process,
                       # from _T0: imports, JIT compile, training and param
                       # save all included — eval.py kills the process at
                       # this wall; return params with margin to spare
MAX_ITER   = 1_000_000 # safety cap on the loop (not expected to bind)

# --- Eval gate: full fidelity only under task/eval.py ---
if _os.environ.get("FORGE_EVAL_TOKEN") != "task/eval.py":
    TRAIN_TIME = 10.0                                # off-eval runs are smoke:
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")   # CPU, 10 s
# ═══════════════════════════════════════════════════════════════

import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


def _locate(name: str) -> Path:
    """Find task/<name> by walking up from cwd (eval.py runs from root)."""
    here = Path.cwd()
    for base in [here, *here.parents]:
        cand = base / "task" / name
        if cand.is_file():
            return cand
    return Path("task") / name


def inlet_centreline(t):
    """Driving speed U(s) at the channel centreline, s = t + T_START.

    The task clock is offset from the benchmark's: the window opens 4.8 s into
    a run that started from rest, which is why the offset appears here rather
    than being folded away.
    """
    return U_PEAK * jnp.sin(jnp.pi * (t + T_START) / T_DRIVE)


def inlet_profile(t, y):
    """Parabolic inlet of instantaneous centreline speed U(t), mean 2/3 U(t)."""
    return 4.0 * inlet_centreline(t) * y * (Y_MAX - y) / (Y_MAX ** 2)


# ══════════ Vanilla PINN: 5 × 100 tanh, output (u, v, p) ══════════

class PINN(nn.Module):
    """5 hidden layers × 100 nodes, tanh activation everywhere, He-uniform
    init, linear output of 3 channels (u, v, p). No feature mapping, no
    shared-trunk / split-head structure — a single MLP over (t, x, y)."""
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, txy):
        # txy: [N, 3]
        kinit = jax.nn.initializers.he_uniform()
        h = txy
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)  # → [N, 3]: (u, v, p)


# ══════════ Data: analytic collocation + analytic BC, IC from file ══════════

def _build_dataset():
    """Collocation grid, the four boundary sets, and the initial condition.

    The geometry is a circle in a rectangle, so every boundary set is
    generated analytically here — there is no shipped geometry file. The
    initial state is the only field data read.

    Returns:
      pts_col : [Nc, 3] (t, x, y) interior collocation, cylinder removed
      pts_ic  : [Ni, 3] (0, x, y) initial-condition coordinates
      uv_ic   : [Ni, 2] (u, v) at t = 0
      pts_in  : [Nb, 3] inlet   x = X_MIN   -> u = inlet_profile(t, y), v = 0
      pts_out : [Nb, 3] outlet  x = X_MAX   -> p = 0
      pts_wall: [Nw, 3] channel walls y = Y_MIN, Y_MAX -> u = v = 0
      pts_cyl : [Nk, 3] cylinder surface              -> u = v = 0
    """
    # ---- interior collocation: uniform grid minus the cylinder ----
    nt, nx, ny = 26, 221, 42
    ts = np.linspace(T_MIN, T_MAX, nt)[1:]        # t = 0 is the IC plane
    xs = np.linspace(X_MIN, X_MAX, nx)[1:-1]      # x faces are BC planes
    ys = np.linspace(Y_MIN, Y_MAX, ny)[1:-1]      # y faces are the walls
    T, X, Y = np.meshgrid(ts, xs, ys, indexing="ij")
    inside = (X - CYL_X) ** 2 + (Y - CYL_Y) ** 2 <= CYL_R ** 2
    pts_col = np.stack([T[~inside], X[~inside], Y[~inside]], axis=1).astype(np.float32)

    # ---- initial condition (the ONLY field data we may read) ----
    ic = np.loadtxt(_locate("cyl_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, y_ic, u_ic, v_ic = ic[:, 0], ic[:, 1], ic[:, 2], ic[:, 3]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic, y_ic], axis=1).astype(np.float32)
    uv_ic = np.stack([u_ic, v_ic], axis=1).astype(np.float32)

    # ---- boundaries ----
    nb, nw, nk = 64, 256, 128
    tb = np.linspace(T_MIN, T_MAX, nb)

    yb = np.linspace(Y_MIN, Y_MAX, nb)
    TB, YB = np.meshgrid(tb, yb, indexing="ij")
    pts_in = np.stack([TB.ravel(), np.full(TB.size, X_MIN), YB.ravel()], 1)
    pts_out = np.stack([TB.ravel(), np.full(TB.size, X_MAX), YB.ravel()], 1)

    xw = np.linspace(X_MIN, X_MAX, nw)
    TW, XW = np.meshgrid(tb, xw, indexing="ij")
    pts_wall = np.concatenate([
        np.stack([TW.ravel(), XW.ravel(), np.full(TW.size, Y_MIN)], 1),
        np.stack([TW.ravel(), XW.ravel(), np.full(TW.size, Y_MAX)], 1)])

    th = np.linspace(0.0, 2.0 * np.pi, nk, endpoint=False)
    TC, TH = np.meshgrid(tb, th, indexing="ij")
    pts_cyl = np.stack([TC.ravel(),
                        CYL_X + CYL_R * np.cos(TH.ravel()),
                        CYL_Y + CYL_R * np.sin(TH.ravel())], 1)

    return (jnp.array(pts_col), jnp.array(pts_ic), jnp.array(uv_ic),
            jnp.array(pts_in.astype(np.float32)),
            jnp.array(pts_out.astype(np.float32)),
            jnp.array(pts_wall.astype(np.float32)),
            jnp.array(pts_cyl.astype(np.float32)))


# ══════════ Training hyperparameters (intentionally minimal) ══════════

LR        = 1e-3      # fixed, no schedule
WEIGHT_IC = 1.0       # vanilla: no loss re-weighting
WEIGHT_BC = 1.0
BS_COL    = 1000      # collocation points per minibatch
BS_IC     = 128       # IC points per minibatch
BS_BC     = 128       # points per boundary set per minibatch
SAVE_MARGIN_S = 5.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


# ══════════ Loss: PDE residual via autograd, IC + BC L2 ══════════

def _make_train(rng_init):
    """Build the JIT'd minibatch + residual + loss + update closures."""
    (pts_col, pts_ic, uv_ic,
     pts_in, pts_out, pts_wall, pts_cyl) = _build_dataset()
    n_col, n_ic = len(pts_col), len(pts_ic)
    n_in, n_out, n_wall, n_cyl = (len(pts_in), len(pts_out),
                                  len(pts_wall), len(pts_cyl))

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def predict_one(params_flat, txy):
        return model.apply(unravel(params_flat), txy[None, :])[0]   # [3]

    d1_fn = jacfwd(predict_one, argnums=1)                          # [3, 3]
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)       # [3, 3, 3]

    def residual_per_point(params_flat, txy):
        uvp = predict_one(params_flat, txy)
        u, v = uvp[0], uvp[1]
        d1 = d1_fn(params_flat, txy)
        u_t, u_x, u_y = d1[0, 0], d1[0, 1], d1[0, 2]
        v_t, v_x, v_y = d1[1, 0], d1[1, 1], d1[1, 2]
        p_x, p_y = d1[2, 1], d1[2, 2]
        d2 = d2_fn(params_flat, txy)
        u_xx, u_yy = d2[0, 1, 1], d2[0, 2, 2]
        v_xx, v_yy = d2[1, 1, 1], d2[1, 2, 2]
        r_c  = u_x + v_y
        r_mx = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        r_my = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
        return r_c ** 2 + r_mx ** 2 + r_my ** 2

    residual_batch = vmap(residual_per_point, in_axes=(None, 0))
    predict_batch = vmap(predict_one, in_axes=(None, 0))

    def loss_fn(params_flat, col, ic_txy, ic_uv, b_in, b_out, b_wall, b_cyl):
        pde_loss = jnp.mean(residual_batch(params_flat, col))

        ic = predict_batch(params_flat, ic_txy)
        ic_loss = jnp.mean((ic[:, 0] - ic_uv[:, 0]) ** 2
                           + (ic[:, 1] - ic_uv[:, 1]) ** 2)

        fin = predict_batch(params_flat, b_in)
        l_in = jnp.mean((fin[:, 0] - inlet_profile(b_in[:, 0], b_in[:, 2])) ** 2
                        + fin[:, 1] ** 2)
        fout = predict_batch(params_flat, b_out)
        l_out = jnp.mean(fout[:, 2] ** 2)
        fw = predict_batch(params_flat, b_wall)
        l_wall = jnp.mean(fw[:, 0] ** 2 + fw[:, 1] ** 2)
        fc = predict_batch(params_flat, b_cyl)
        l_cyl = jnp.mean(fc[:, 0] ** 2 + fc[:, 1] ** 2)

        bc_loss = l_in + l_out + l_wall + l_cyl
        return pde_loss + WEIGHT_IC * ic_loss + WEIGHT_BC * bc_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, opt_state, keys):
        k1, k2, k3, k4, k5, k6 = keys
        i_ic = random.choice(k2, n_ic, (BS_IC,))
        loss, grads = loss_and_grad(
            params_flat,
            pts_col[random.choice(k1, n_col, (BS_COL,))],
            pts_ic[i_ic], uv_ic[i_ic],
            pts_in[random.choice(k3, n_in, (BS_BC,))],
            pts_out[random.choice(k4, n_out, (BS_BC,))],
            pts_wall[random.choice(k5, n_wall, (BS_BC,))],
            pts_cyl[random.choice(k6, n_cyl, (BS_BC,))],
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params_flat, updates), opt_state, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: [BS, 3] with columns (t, x, y). Returns {"u","v","p"} each [BS]."""
    out = _PREDICT_MODEL.apply(params, X)   # [BS, 3]
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam training; the whole process finishes within
    TRAIN_TIME s of _T0. Returns (params_pytree, step_count)."""
    params_flat, opt_state, update, unravel, rng_local = _make_train(rng)

    # Warmup step JIT-compiles the update; counted.
    keys = random.split(rng_local, 7)
    rng_local = keys[0]
    params_flat, opt_state, loss = update(params_flat, opt_state, tuple(keys[1:]))
    step_count = 1
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        keys = random.split(rng_local, 7)
        rng_local = keys[0]
        params_flat, opt_state, loss = update(params_flat, opt_state, tuple(keys[1:]))
        step_count = it + 1
        last_loss = float(loss)
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(params_flat), it, last_loss)

    return unravel(params_flat), step_count
