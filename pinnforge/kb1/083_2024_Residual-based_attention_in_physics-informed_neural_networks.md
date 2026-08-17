---
slot: 83
title: "Residual-based attention in physics-informed neural networks"
authors: [Sokratis J. Anagnostopoulos, Juan Diego Toscano, Nikolaos Stergiopulos, George Em Karniadakis]
year: 2024
venue: "Computer Methods in Applied Mechanics and Engineering 421 (2024) 116805"
gitrepo: "https://github.com/soanagno/rba-pinns"
doi: "10.1016/j.cma.2024.116805"
---

## TL;DR
RBA-PINNs maintain per-collocation-point weights that are an **exponentially-weighted moving average of the normalised residual** - a cheap, gradient-free attention mask that focuses the optimiser on persistently hard points. The mask is bounded (no exploding multipliers) and adds essentially zero overhead. Combined with modified-MLP, weight normalisation and exact BC enforcement, it reaches state-of-the-art L2 errors on Allen-Cahn, Helmholtz and AIV inverse problems.

## Problem
Self-Adaptive (SA-PINN) and adversarial point-weighting methods need auxiliary networks or adversarial gradients, are expensive, and can grow unboundedly. NTK / gradient-based balancing requires extra autograd passes. A simple, stable, gradient-less alternative is desirable.

## Method
For loss term `j` and collocation point `i`, residual `e_{i,j}`, maintain a weight `lambda_{i,j}` updated like Adam's EMA but driven by the *normalised* point residual:
$$ \lambda_{i,j}^{k+1} \leftarrow \gamma \lambda_{i,j}^{k} + \eta^{*}\frac{|e_{i,j}|}{\|e_{\cdot,j}\|_\infty} $$
Bounded by `(0, eta*/(1-gamma)]`. Loss becomes
$$ \mathcal{L} = \sum_{j\in\{r,ic,bc,d\}} \langle (\lambda^{*}_{i,j}\,e_{i,j})^2 \rangle_i $$
with optional offset `lambda_o`. Combined enhancements: (i) modified-MLP backbone (input encoders U,V gating each layer), (ii) weight-normalisation (`theta = g v/||v||`), (iii) exact Dirichlet (`u = g(x) + phi(x) u_NN(x)`) and exact periodic via Fourier features.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class ModMLP(nn.Module):
    H: int = 256
    depth: int = 8
    d_out: int = 1
    @nn.compact
    def __call__(self, x):
        U = nn.tanh(nn.Dense(self.H, name="U")(x))
        V = nn.tanh(nn.Dense(self.H, name="V")(x))
        h = nn.tanh(nn.Dense(self.H, name="L0")(x))
        for i in range(1, self.depth):
            z = nn.tanh(nn.Dense(self.H, name=f"L{i}")(h))
            h = (1 - z) * U + z * V
        return nn.Dense(self.d_out)(h)

# RBA weights per loss term (stored outside params, treated as constants in loss)
lam = {"r": jnp.zeros(N_r), "bc": jnp.zeros(N_bc), "ic": jnp.zeros(N_ic)}
gamma, eta_star, lam_o = 0.999, 0.01, 0.0

def residuals(params, x):
    return {
        "r" : pde_residual(params, x["r"]),
        "bc": ModMLP().apply(params, x["bc"]) - g_bc,
        "ic": ModMLP().apply(params, x["ic"]) - u0,
    }

def loss_fn(params, x, lam, lam_o):
    res = residuals(params, x)
    return sum(jnp.mean((jax.lax.stop_gradient(lam[j] + lam_o) * res[j].reshape(-1))**2)
               for j in res)

optimizer = optax.adam(1e-3); opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, lam, x, k):
    # RBA EMA update (gradient-free, on detached residuals)
    res = residuals(params, x)
    new_lam = {}
    for j, e in res.items():
        e_abs   = jnp.abs(e).reshape(-1)
        eta_eff = jnp.where(k == 1, 1.0, eta_star)
        new_lam[j] = gamma * lam[j] + eta_eff * e_abs / (jnp.max(e_abs) + 1e-12)
    grads = jax.grad(loss_fn)(params, x, new_lam, lam_o)
    upd, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, upd), opt_state, new_lam
```

Hyperparameters: ModMLP 8 hidden x 256 with weight-norm; Adam `lr=1e-3` with exponential decay 0.9; `gamma=0.999`, `eta*=1e-2`; `lambda_o=0` for clean problems, small positive for noisy AIV; `w_ic = 100`, `w_r = 1` global weights; 3e5 iterations on Allen-Cahn (`eps=1e-4`); apply RBA only to PDE residual term unless inverse problem.

## Results
On 1-D Allen-Cahn (`eps=1e-4`), RBA achieves relative L2 `2.0e-5` (~1 order better than SA and causal-PINN). On Helmholtz, on the order of `1e-5`. On the brain-perivascular-space AIV inverse Navier-Stokes problem, RBA halves the pressure-prediction error vs vanilla PINN, with negligible additional cost.
