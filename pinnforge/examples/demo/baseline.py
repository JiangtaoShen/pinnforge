"""Vanilla PINN baseline for the multi-scale Poisson demo (tree root).

Contract (do not change):
  - ``predict_fn(params, X) -> {"u"}``
  - ``train(rng, eval_callback=None) -> (params, step_count)``
  - Frozen PDE-CONSTANTS header is byte-identical across all descendants.

Time budget (frozen): ``TRAIN_TIME = 60`` s of wall-clock for the
WHOLE process, anchored at ``_T0`` (process start) — imports, JIT
compilation, training and param save all included; ``eval.py`` kills
the process at this wall. ``train()`` must return params with margin
to spare (this baseline stops ``SAVE_MARGIN_S = 5`` s early).
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# PDE CONSTANTS — DO NOT MODIFY (frozen header, byte-identical to parent)
# ═══════════════════════════════════════════════════════════════

# --- I/O shape ---
INPUT_DIM    = 2
OUTPUT_DIM   = 1
INPUT_NAMES  = ("x", "y")
FIELD_NAMES  = ("u",)

# --- Domain ---
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 1
HAS_SOURCE              = True

# --- Boundary topology ---
BOUNDARY_NAMES = ("top", "bottom", "left", "right")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "box"

# --- Physical parameters ---
import math as _math
K_FINE   = 8.0       # the fine mode is K_FINE times the coarse one
A_FINE   = 0.1       # ... with this amplitude
PI       = _math.pi

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 60.0      # total wall-clock seconds for the WHOLE process,
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
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


def source(x, y):
    """f = -(u_xx + u_yy) for the two-scale solution."""
    s1 = jnp.sin(PI * x) * jnp.sin(PI * y)
    sk = jnp.sin(K_FINE * PI * x) * jnp.sin(K_FINE * PI * y)
    return 2.0 * PI ** 2 * s1 + A_FINE * 2.0 * (K_FINE * PI) ** 2 * sk


# ══════════ Vanilla PINN: 4 × 64 tanh, output u ══════════

class PINN(nn.Module):
    """4 hidden layers × 64 nodes, tanh, Glorot init, linear scalar output.
    No feature mapping, no hard constraints — a single MLP."""
    hidden_width: int = 64
    n_hidden_layers: int = 4

    @nn.compact
    def __call__(self, xy):
        h = xy
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM)(h)  # → [N, 1]


# ══════════ Data: 64×64 interior grid + boundary points ══════════

def _build_dataset():
    """Strict-interior collocation grid and wall points (u = 0 there)."""
    n = 64
    xs = np.linspace(X_MIN, X_MAX, n + 2)[1:-1]
    ys = np.linspace(Y_MIN, Y_MAX, n + 2)[1:-1]
    X, Y = np.meshgrid(xs, ys)
    pts_int = np.stack([X.ravel(), Y.ravel()], axis=1)

    t = np.linspace(0.0, 1.0, 129)
    zeros, ones = np.zeros_like(t), np.ones_like(t)
    pts_bc = np.concatenate([
        np.stack([t, zeros], axis=1), np.stack([t, ones], axis=1),
        np.stack([zeros, t], axis=1), np.stack([ones, t], axis=1),
    ])
    return jnp.array(pts_int.astype(np.float32)), jnp.array(pts_bc.astype(np.float32))


# ══════════ Training hyperparameters (intentionally minimal) ══════════

LR        = 1e-3      # fixed, no schedule
WEIGHT_BC = 1.0       # no PDE-vs-BC re-weighting (vanilla)
BS_INT    = 512       # interior collocation points per minibatch
BS_BC     = 128       # boundary points per minibatch
SAVE_MARGIN_S = 5.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


def _make_train(rng_init):
    pts_int, pts_bc = _build_dataset()
    n_int, n_bc = len(pts_int), len(pts_bc)

    model = PINN()
    key, rng = random.split(rng_init)
    params = model.init(key, jnp.zeros([1, INPUT_DIM]))
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def u_one(params_flat, x, y):
        return model.apply(unravel(params_flat), jnp.stack([x, y])[None, :])[0, 0]

    u_xx = grad(grad(u_one, argnums=1), argnums=1)
    u_yy = grad(grad(u_one, argnums=2), argnums=2)

    def residual(params_flat, xy):
        x, y = xy[0], xy[1]
        return -(u_xx(params_flat, x, y) + u_yy(params_flat, x, y)) - source(x, y)

    residual_batch = vmap(residual, in_axes=(None, 0))

    def loss_fn(params_flat, xy_int, xy_bc):
        pde = jnp.mean(residual_batch(params_flat, xy_int) ** 2)
        u_bc = model.apply(unravel(params_flat), xy_bc)[:, 0]
        return pde + WEIGHT_BC * jnp.mean(u_bc ** 2)

    loss_and_grad = jit(jax.value_and_grad(loss_fn))
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, opt_state, key_pair):
        key1, key2 = key_pair
        idx_int = random.choice(key1, n_int, (BS_INT,))
        idx_bc = random.choice(key2, n_bc, (BS_BC,))
        loss, grads = loss_and_grad(params_flat, pts_int[idx_int], pts_bc[idx_bc])
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params_flat, updates), opt_state, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: jnp.ndarray of shape [BS, 2] with columns (x, y). Returns {"u": [BS]}."""
    return {"u": _PREDICT_MODEL.apply(params, X)[:, 0]}


def train(rng, eval_callback: Callable[[Any, int, float], None] | None = None):
    """Wall-deadline Adam training; stops SAVE_MARGIN_S before the wall.

    Returns ``(params, step_count)`` — ``params`` is a Flax pytree.
    """
    params_flat, opt_state, update, unravel, rng_local = _make_train(rng)

    # Warmup step: the first update() call JIT-compiles the whole training
    # step; float(loss) blocks until it finishes. It is a real step (counted).
    key1, key2, rng_local = random.split(rng_local, 3)
    params_flat, opt_state, loss = update(params_flat, opt_state, (key1, key2))
    step_count = 1
    last_loss = float(loss)

    deadline = _T0 + TRAIN_TIME - SAVE_MARGIN_S
    for it in range(1, MAX_ITER):
        if time.time() > deadline:
            break
        key1, key2, rng_local = random.split(rng_local, 3)
        params_flat, opt_state, loss = update(params_flat, opt_state, (key1, key2))
        step_count = it + 1
        if eval_callback is not None and (it % 1000 == 0):
            last_loss = float(loss)
            eval_callback(unravel(params_flat), it, last_loss)

    return unravel(params_flat), step_count
