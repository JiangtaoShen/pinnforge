---
slot: 72
title: "Adaptive Training of Grid-Dependent Physics-Informed Kolmogorov-Arnold Networks"
authors: [Spyros Rigas, Michalis Papachristou, Theofilos Papadopoulos, Fotios Anagnostopoulos, Georgios Alexandridis]
year: 2024
venue: "IEEE Access (arXiv:2407.17611)"
gitrepo: ""
---

## TL;DR
Physics-Informed Kolmogorov-Arnold Networks (PIKANs) replace MLP activations by grid-dependent learnable B-spline basis functions; training is stabilised by an adaptive **state-transition** that interpolates Adam moments after grid extension and by adaptive loss re-weighting / collocation resampling, yielding up to 84× speedup over `pykan` and 43% L2-error reduction.

## Problem
Vanilla PIKANs trained with `pykan` are slow (B-spline activations, 200 ms/epoch) and exhibit sharp loss spikes whenever the grid is extended or adapted because (i) B-spline knots move abruptly and (ii) the Adam optimiser state is reset, erasing first/second moment EMAs.

## Method
A KAN layer of shape `[n_in, n_out]` with grid size `G` and spline order `k` has activations `phi(x) = c_r * silu(x) + sum_i c_i B_i(x)`, where `B_i` are B-splines on a knot grid. Training consists of (a) periodic grid updates `G = ge*G_uniform + (1-ge)*G_adaptive`, (b) grid extension `G -> G'`, (c) adaptive state transition copying `c_r,c_B` moments and linearly interpolating moments for the new `c_i`, (d) optional Adam learning-rate reduction post-update, (e) global-NTK-style adaptive loss weights and RAD-type collocation resampling.

Loss: `L = w_f L_f + sum_k w_bk L_bk` with `L_f = (1/N_f) sum ||F(u(x_i;theta)) - f(x_i)||^2`.

This paper's reference implementation is `jaxKAN`, which is already JAX-native.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class KANLayer(nn.Module):
    n_in:  int
    n_out: int
    G:     int = 4
    k:     int = 3
    grid_range: tuple = (-1.0, 1.0)

    @nn.compact
    def __call__(self, x):                             # x: (B, n_in)
        grid = self.variable("grid", "knots",
                             lambda: jnp.broadcast_to(
                                 jnp.linspace(*self.grid_range, self.G + 1),
                                 (self.n_in, self.G + 1)))
        c    = self.param("c",   nn.initializers.normal(stddev=0.1),
                          (self.n_in, self.n_out, self.G + self.k))
        c_r  = self.param("c_r", nn.initializers.ones, (self.n_in, self.n_out))
        c_B  = self.param("c_B", nn.initializers.ones, (self.n_in, self.n_out))
        B = b_splines(x, grid.value, self.k)           # (B, n_in, G+k); Cox-de Boor
        r = jax.nn.silu(x)[..., None]                  # (B, n_in, 1)
        phi = c_r * r + jnp.einsum("bik,iok->bio", B, c)
        return phi.sum(axis=-2)                        # (B, n_out)

def adam_state_transition(opt_state, c_path, old_grid, new_grid):
    """Linearly interpolate exp_avg / exp_avg_sq for c along basis axis."""
    # opt_state is optax.ScaleByAdamState (nested in optax.GradientTransformation)
    def interp(t):
        new = interp1d_along_basis(t, old_grid, new_grid)   # user-supplied
        return new
    # Pseudocode — apply optax.tree_utils to drill into the moments
    new_state = jax.tree_util.tree_map_with_path(
        lambda path, leaf: interp(leaf) if c_path in path else leaf, opt_state)
    return new_state

def extend_grid(params, layer_name, new_G):
    # least-squares refit of c on the new grid (vmapped)
    params[layer_name]["c"] = least_squares_refit(params, layer_name, new_G)
    return params

opt = optax.adam(3e-3)
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, X_r, X_b):
    def loss(p):
        return w_f * Lf(p, X_r) + sum(w_b * Lb(p, X_b) for Lb in bcs)
    grads = jax.grad(loss)(params)
    upd, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, upd), opt_state

for epoch in range(N):
    if epoch in update_epochs:
        adapt_grid(model_state)                    # eq. (11) ge-mixed
        if epoch in extend_epochs:
            params = extend_grid(params, layer_name, new_G)
            opt_state = adam_state_transition(opt_state, c_path,
                                              old_grid, new_grid)
            opt = optax.adam(opt.hyperparams["learning_rate"] * 0.5)
    params, opt_state = step(params, opt_state, X_r, X_b)
```

Hyperparameters: `k=3`, initial `G=3..5`, extend by `+5..+10` every ~`1e3` epochs, `ge=0.0..0.2`, Adam `lr=3e-3`, post-extension LR factor `0.5`.

## Results
On 4 benchmark PDEs (Diffusion, Helmholtz, Burgers, Allen-Cahn) `jaxKAN` is two orders of magnitude faster per epoch than `pykan` (~3 vs ~200 ms). Adaptive transition removes loss spikes and reduces final relative L2 by up to 43%, matching or beating MLP-PINNs that use 8.5× more parameters.
