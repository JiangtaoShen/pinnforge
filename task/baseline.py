"""Vanilla PINN baseline for the KS-chaotic benchmark (tree root).

Contract (do not change):
  - ``predict_fn(params, X) -> {"u"}``
  - ``train(rng, eval_callback=None) -> (params, step_count)``
  - Frozen PDE-CONSTANTS header is byte-identical across all descendants.

Time budget (frozen): ``TRAIN_TIME = 300`` s of wall-clock for the
WHOLE process, anchored at ``_T0`` (process start) — imports, JIT
compilation, training and param save all included; ``eval.py`` kills
the process at this wall. ``train()`` must return params with margin
to spare (this baseline stops ``SAVE_MARGIN_S = 5`` s early).

Data discipline: the ONLY field data this baseline (and any descendant)
may read is the initial condition ``task/ks_ic.csv`` (u at t=0). Never
read ``task/ref_data.csv`` — it is the scoring truth, used by
``task/eval.py`` for scoring alone.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 2
OUTPUT_DIM   = 1
INPUT_NAMES  = ("t", "x")
FIELD_NAMES  = ("u",)

# --- Domain ---
T_MIN, T_MAX = 0.0, 0.4
X_MIN, X_MAX = 0.0, 6.283185307179586   # 2*pi
HAS_TIME = True

# --- Equation structure ---
TIME_DERIV_ORDER        = 1
MAX_SPATIAL_DERIV_ORDER = 4   # u_xxxx
N_RESIDUAL_COMPONENTS   = 1
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("x_lo", "x_hi")
PERIODIC_AXES  = ("x",)       # u(t, X_MIN) = u(t, X_MAX)
GEOMETRY_KIND  = "periodic_strip"

# --- Physical parameters (scaled KS: u_t + V1 u u_x + V2 u_xx + V3 u_xxxx = 0) ---
V1 = 100.0 / 16.0        # 6.25
V2 = 100.0 / 16.0**2     # 0.390625
V3 = 100.0 / 16.0**4     # ~1.5259e-3

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
from jax import grad, jit, random, vmap
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


# ══════════ Vanilla PINN: 5 × 100 tanh, output u ══════════

class PINN(nn.Module):
    """5 hidden layers × 100 nodes, tanh everywhere, He-uniform init,
    linear scalar output u. A single plain MLP — no feature mapping, no
    skip connections, no special activations."""
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, tx):
        # tx: [N, 2] with columns (t, x)
        kinit = jax.nn.initializers.he_uniform()
        h = tx
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)  # → [N, 1]: u


# ══════════ Data: analytic collocation + periodic BC, IC from file ══════════

def _build_dataset():
    """Collocation grid + IC points (from ks_ic.csv) + periodic-BC t-samples.

    Returns:
      pts_col : [Nc, 2] interior collocation coordinates (PDE residual)
      pts_ic  : [Ni, 2] (t=0, x) coordinates for the initial condition
      u_ic    : [Ni]    IC target values u(0, x)
      t_bc    : [Nb]    time samples for the periodic constraint
                        u(t, X_MIN) = u(t, X_MAX)
    """
    # collocation: uniform grid, strict interior in time (t=0 IC line
    # excluded — the PDE residual is not enforced on the supervised
    # initial condition), x on the half-open ring [0, 2pi).
    nt, nx = 101, 512
    ts = np.linspace(T_MIN, T_MAX, nt)[1:]
    xs = np.linspace(X_MIN, X_MAX, nx, endpoint=False)
    T, X = np.meshgrid(ts, xs, indexing="ij")
    pts_col = np.stack([T.ravel(), X.ravel()], axis=1).astype(np.float32)

    # initial condition (the ONLY field data we may read)
    ic = np.loadtxt(_locate("ks_ic.csv"), delimiter=",", skiprows=1, dtype=np.float64)
    x_ic, u_ic = ic[:, 0], ic[:, 1]
    pts_ic = np.stack([np.zeros_like(x_ic), x_ic], axis=1).astype(np.float32)

    # periodic BC: time samples at which u(t,0) and u(t,2pi) must agree
    t_bc = np.linspace(T_MIN, T_MAX, nt).astype(np.float32)

    return (
        jnp.array(pts_col),
        jnp.array(pts_ic),
        jnp.array(u_ic.astype(np.float32)),
        jnp.array(t_bc),
    )


# ══════════ Training hyperparameters (intentionally minimal) ══════════

LR        = 1e-3      # fixed, no schedule
WEIGHT_IC = 1.0       # vanilla: no loss re-weighting
WEIGHT_BC = 1.0
BS_COL    = 1000      # collocation points per minibatch
BS_IC     = 128       # IC points per minibatch
BS_BC     = 128       # periodic-BC time samples per minibatch
SAVE_MARGIN_S = 5.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


# ══════════ Loss: KS residual via autograd, IC + periodic-BC L2 ══════════

def _make_train(rng_init):
    """Build the JIT'd minibatch + residual + loss + update closures."""
    pts_col, pts_ic, u_ic, t_bc = _build_dataset()
    n_col, n_ic, n_bc = len(pts_col), len(pts_ic), len(t_bc)

    model = PINN()
    key, rng = random.split(rng_init)
    dummy = jnp.zeros([1, INPUT_DIM])
    params = model.init(key, dummy)
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    # scalar u(t, x) for one point
    def u_scalar(params_flat, t, x):
        return model.apply(unravel(params_flat), jnp.stack([t, x])[None, :])[0, 0]

    # KS residual at one point: u_t + V1 u u_x + V2 u_xx + V3 u_xxxx
    u_t_fn   = grad(u_scalar, argnums=1)
    u_x_fn   = grad(u_scalar, argnums=2)
    u_xx_fn  = grad(u_x_fn,  argnums=2)
    u_xxx_fn = grad(u_xx_fn, argnums=2)
    u_xxxx_fn = grad(u_xxx_fn, argnums=2)

    def residual_per_point(params_flat, tx):
        t, x = tx[0], tx[1]
        u     = u_scalar(params_flat, t, x)
        u_t   = u_t_fn(params_flat, t, x)
        u_x   = u_x_fn(params_flat, t, x)
        u_xx  = u_xx_fn(params_flat, t, x)
        u_xxxx = u_xxxx_fn(params_flat, t, x)
        r = u_t + V1 * u * u_x + V2 * u_xx + V3 * u_xxxx
        return r ** 2

    residual_batch = vmap(residual_per_point, in_axes=(None, 0))
    u_batch = vmap(u_scalar, in_axes=(None, 0, 0))

    def loss_fn(params_flat, col, ic_xy, ic_u, bc_t):
        # PDE residual on interior collocation
        pde_loss = jnp.mean(residual_batch(params_flat, col))

        # initial condition
        u_pred_ic = u_batch(params_flat, ic_xy[:, 0], ic_xy[:, 1])
        ic_loss = jnp.mean((u_pred_ic - ic_u) ** 2)

        # periodic BC: u(t, X_MIN) == u(t, X_MAX)
        u_lo = u_batch(params_flat, bc_t, jnp.full_like(bc_t, X_MIN))
        u_hi = u_batch(params_flat, bc_t, jnp.full_like(bc_t, X_MAX))
        bc_loss = jnp.mean((u_lo - u_hi) ** 2)

        return pde_loss + WEIGHT_IC * ic_loss + WEIGHT_BC * bc_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))

    optimizer = optax.adam(learning_rate=LR)   # fixed LR, no schedule
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, opt_state, keys):
        k_col, k_ic, k_bc = keys
        col = pts_col[random.choice(k_col, n_col, (BS_COL,))]
        ic_idx = random.choice(k_ic, n_ic, (BS_IC,))
        bc_t = t_bc[random.choice(k_bc, n_bc, (BS_BC,))]
        loss, grads = loss_and_grad(params_flat, col, pts_ic[ic_idx], u_ic[ic_idx], bc_t)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_flat, updates)
        return new_params, opt_state, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: jnp.ndarray of shape [BS, 2] with columns (t, x).
    Returns dict {"u": [BS]}.
    """
    out = _PREDICT_MODEL.apply(params, X)   # [BS, 1]
    return {"u": out[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam training. The whole process — imports, JIT
    compilation, training, param save — must finish within TRAIN_TIME
    seconds of ``_T0`` (process start); the loop stops SAVE_MARGIN_S
    early so the caller can device_get + pickle before the wall.

    Args:
        rng: jax.random.PRNGKey — passed to ``_make_train`` for deterministic init.
        eval_callback: optional ``callback(params, step, loss)``, fires every
             1000 steps with the unraveled (pytree) params.

    Returns:
        (params, step_count) — ``params`` is a Flax pytree.
    """
    params_flat, opt_state, update, unravel, rng_local = _make_train(rng)

    # Warmup step: the first update() call JIT-compiles the whole training
    # step; float(loss) blocks until it finishes. It is a real step (counted).
    keys = random.split(rng_local, 3)
    rng_local = random.split(rng_local)[0]
    params_flat, opt_state, loss = update(params_flat, opt_state, keys)
    step_count = 1
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        rng_local, sub = random.split(rng_local)
        keys = random.split(sub, 3)
        params_flat, opt_state, loss = update(params_flat, opt_state, keys)
        step_count = it + 1
        last_loss = float(loss)
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(params_flat), it, last_loss)

    return unravel(params_flat), step_count
