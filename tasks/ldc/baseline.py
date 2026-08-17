"""Vanilla PINN baseline for NS-LDC Re=3200 (tree root).

Contract (do not change):
  - ``predict_fn(params, X) -> {"u","v","p"}``
  - ``train(rng, eval_callback=None) -> (params, step_count)``
  - Frozen PDE-CONSTANTS header is byte-identical across all descendants.

Time budget (frozen): ``TRAIN_TIME = 150`` s of wall-clock for the
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
OUTPUT_DIM   = 3
INPUT_NAMES  = ("x", "y")
FIELD_NAMES  = ("u", "v", "p")

# --- Domain ---
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
HAS_TIME = False

# --- Equation structure ---
TIME_DERIV_ORDER        = 0
MAX_SPATIAL_DERIV_ORDER = 2
N_RESIDUAL_COMPONENTS   = 3   # ru, rv, continuity
HAS_SOURCE              = False

# --- Boundary topology ---
BOUNDARY_NAMES = ("top", "bottom", "left", "right")
PERIODIC_AXES  = ()
GEOMETRY_KIND  = "box"

# --- Physical parameters ---
RE     = 3200.0      # NS-LDC Re=3200; harder regime than Re=1000
NU     = 1.0 / RE
U_LID  = 1.0

# --- Training budget ---
import os as _os, time as _time
_T0 = float(_os.environ.get("FORGE_T0", _time.time()))  # process-start wall anchor
TRAIN_TIME = 150.0     # total wall-clock seconds for the WHOLE process,
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
from jax import jacfwd, jit, random, vmap
import jax.flatten_util
import numpy as np
import optax
from flax import linen as nn


# ══════════ Vanilla PINN: 5 × 100 tanh, output (u, v, p) ══════════

class PINN(nn.Module):
    """5 hidden layers × 100 nodes, tanh activation everywhere, He-uniform
    init, linear output of 3 channels (u, v, p). No feature mapping, no
    shared-trunk / split-head structure — a single MLP."""
    hidden_width: int = 100
    n_hidden_layers: int = 5

    @nn.compact
    def __call__(self, xy):
        # xy: [N, 2]
        kinit = jax.nn.initializers.he_uniform()
        h = xy
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_width, kernel_init=kinit)(h)
            h = nn.tanh(h)
        return nn.Dense(OUTPUT_DIM, kernel_init=kinit)(h)  # → [N, 3]: (u, v, p)


# ══════════ Data: synthetic 100×100 grid + analytic BC ══════════

def _build_dataset():
    """100×100 uniform grid over [0,1]^2.

    Returns three disjoint point sets:

    * ``pts_int``: strictly INTERIOR collocation points for the PDE
      residual loss (corners and walls both excluded). The PDE is not
      enforced on Dirichlet boundaries in the standard PINN formulation;
      mixing BC points into the residual loss is statistically wasteful
      and conceptually wrong.
    * ``pts_bc``: boundary (wall) points, corners excluded, for the BC loss.
    * BC values: u = U_LID on top wall, u=0 elsewhere; v=0 on all walls.
    """
    nx, ny = 100, 100
    xs = np.linspace(X_MIN, X_MAX, nx)
    ys = np.linspace(Y_MIN, Y_MAX, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.flatten(), Y.flatten()], axis=1)

    on_wall = (
        (pts[:, 0] == X_MIN) | (pts[:, 0] == X_MAX)
        | (pts[:, 1] == Y_MIN) | (pts[:, 1] == Y_MAX)
    )
    on_corner = (
        ((pts[:, 0] == X_MIN) & (pts[:, 1] == Y_MAX))
        | ((pts[:, 0] == X_MAX) & (pts[:, 1] == Y_MAX))
        | ((pts[:, 0] == X_MIN) & (pts[:, 1] == Y_MIN))
        | ((pts[:, 0] == X_MAX) & (pts[:, 1] == Y_MIN))
    )

    # pts_int: walls (incl. corners) excluded → strict interior.
    pts_int = pts[~on_wall]
    # pts_bc: walls minus corners. (Corners are excluded because the LDC
    # geometry has a discontinuous BC at the two top corners — u=U_LID on
    # the lid but u=0 on the adjacent side walls — so the BC value there
    # is ambiguous.)
    pts_bc = pts[on_wall & ~on_corner]

    u_bc_vals = np.where(pts_bc[:, 1] == Y_MAX, U_LID, 0.0).astype(np.float32)
    v_bc_vals = np.zeros(len(pts_bc), dtype=np.float32)

    return (
        jnp.array(pts_int.astype(np.float32)),
        jnp.array(pts_bc.astype(np.float32)),
        jnp.array(u_bc_vals),
        jnp.array(v_bc_vals),
    )


# ══════════ Training hyperparameters (intentionally minimal) ══════════

LR        = 1e-3      # fixed, no schedule
WEIGHT_BC = 1.0       # default — no PDE-vs-BC re-weighting (vanilla)
BS_INT    = 350       # interior collocation points per minibatch
BS_BC     = 50        # boundary points per minibatch (matches batch_size=400)
SAVE_MARGIN_S = 5.0   # stop the loop this early so device_get + pickling
                      # finish before eval.py's TRAIN_TIME wall


# ══════════ Loss: PDE residual via autograd, BC L2 ══════════

def _make_train(rng_init):
    """Build the JIT'd minibatch + residual + loss + update closures."""
    pts_int, pts_bc, u_bc, v_bc = _build_dataset()
    n_int, n_bc = len(pts_int), len(pts_bc)

    model = PINN()
    key, rng = random.split(rng_init)
    dummy = jnp.zeros([1, INPUT_DIM])
    params = model.init(key, dummy)
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    # Single-point forward → returns [3]: (u, v, p)
    def predict_one(params_flat, xy):
        return model.apply(unravel(params_flat), xy[None, :])[0]

    # First derivatives: jacobian wrt xy → shape [3, 2]
    # rows = (u, v, p); cols = (x, y) partials
    d1_fn = jacfwd(predict_one, argnums=1)

    # Second derivatives: hessian-like, shape [3, 2, 2]
    # [field_idx, i, j] = ∂²(u/v/p)/(∂xy[i]∂xy[j])
    d2_fn = jacfwd(jacfwd(predict_one, argnums=1), argnums=1)

    def residual_per_point(params_flat, xy):
        uvp = predict_one(params_flat, xy)
        u, v, p = uvp[0], uvp[1], uvp[2]

        d1 = d1_fn(params_flat, xy)            # [3, 2]
        u_x, u_y = d1[0, 0], d1[0, 1]
        v_x, v_y = d1[1, 0], d1[1, 1]
        p_x, p_y = d1[2, 0], d1[2, 1]

        d2 = d2_fn(params_flat, xy)            # [3, 2, 2]
        u_xx, u_yy = d2[0, 0, 0], d2[0, 1, 1]
        v_xx, v_yy = d2[1, 0, 0], d2[1, 1, 1]

        # NS residuals
        r_c  = u_x + v_y
        r_mx = u * u_x + v * u_y + p_x - (1.0 / RE) * (u_xx + u_yy)
        r_my = u * v_x + v * v_y + p_y - (1.0 / RE) * (v_xx + v_yy)

        return r_c ** 2 + r_mx ** 2 + r_my ** 2

    # Vectorize over the interior batch
    residual_batch = vmap(residual_per_point, in_axes=(None, 0))

    def loss_fn(params_flat, xy_int, xy_bc_batch, u_bc_batch, v_bc_batch):
        # PDE loss on interior collocation
        pde_per_pt = residual_batch(params_flat, xy_int)
        pde_loss   = jnp.mean(pde_per_pt)

        # BC loss on boundary (only u, v supervised — pressure is gauge-free)
        bc_uvp = vmap(predict_one, in_axes=(None, 0))(params_flat, xy_bc_batch)
        u_pred, v_pred = bc_uvp[:, 0], bc_uvp[:, 1]
        bc_loss = jnp.mean((u_pred - u_bc_batch) ** 2 + (v_pred - v_bc_batch) ** 2)

        return pde_loss + WEIGHT_BC * bc_loss

    loss_and_grad = jit(jax.value_and_grad(loss_fn))

    optimizer = optax.adam(learning_rate=LR)   # fixed LR, no schedule
    opt_state = optimizer.init(flat_params)

    @jit
    def update(params_flat, opt_state, key_pair):
        key1, key2 = key_pair
        idx_int = random.choice(key1, n_int, (BS_INT,))
        idx_bc  = random.choice(key2, n_bc,  (BS_BC,))
        xy_int_batch = pts_int[idx_int]
        xy_bc_batch  = pts_bc[idx_bc]
        ub = u_bc[idx_bc]
        vb = v_bc[idx_bc]
        loss, grads = loss_and_grad(params_flat, xy_int_batch, xy_bc_batch, ub, vb)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params_flat, updates)
        return new_params, opt_state, loss

    return flat_params, opt_state, update, unravel, rng


# ══════════ Public contract ══════════

_PREDICT_MODEL = PINN()


def predict_fn(params, X):
    """X: jnp.ndarray of shape [BS, 2] with columns (x, y).
    Returns dict {"u": [BS], "v": [BS], "p": [BS]}.
    """
    out = _PREDICT_MODEL.apply(params, X)   # [BS, 3]
    return {"u": out[:, 0], "v": out[:, 1], "p": out[:, 2]}


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
        last_loss = float(loss)
        if eval_callback is not None and (it % 1000 == 0):
            eval_callback(unravel(params_flat), it, last_loss)

    return unravel(params_flat), step_count
