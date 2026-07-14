---
slot: 82
title: "PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks"
authors: [Sifan Wang, Bowen Li, Yuhan Chen, Paris Perdikaris]
year: 2024
venue: "Journal of Machine Learning Research (arXiv:2402.00326)"
gitrepo: "https://github.com/PredictiveIntelligenceLab/jaxpi"
---

## TL;DR
Deep MLPs make PINNs *worse*, because Glorot initialisation makes the network *derivatives* almost constant (the residual loss therefore starts in a pathological regime). PirateNets fix this with an **adaptive residual connection** whose gain `alpha` is initialised to 0 (block = identity), so at start the model is a linear combination of Fourier-feature embeddings. As training proceeds, `alpha` grows and the network "deepens". The last layer is also initialised by least-squares to fit known data (initial condition / linearised PDE) - a physics-informed init.

## Problem
For a Tanh-MLP with Glorot init, `du/dx (x) ≈ W^(L+1) W^(L) ... W^(1)` in the linear regime, with `Var(du/dx) ~ 1/d` (depth-independent). Higher-order derivatives behave similarly. Hence the PDE residual `R[u_theta]` and its loss start in a degenerate, near-trivial region; deeper MLPs (18 layers) reach ~100% L2 error on Allen-Cahn.

## Method
**Architecture (PirateNet block):**
$$ \mathbf{f}^{(l)} = \sigma(W_1^{(l)} \mathbf{x}^{(l)} + \mathbf{b}_1^{(l)}),\quad \mathbf{z}_1^{(l)} = \mathbf{f}^{(l)} \odot U + (1-\mathbf{f}^{(l)}) \odot V $$
$$ \mathbf{g}^{(l)} = \sigma(W_2 \mathbf{z}_1 + \mathbf{b}_2),\quad \mathbf{z}_2 = \mathbf{g}\odot U + (1-\mathbf{g})\odot V $$
$$ \mathbf{h}^{(l)} = \sigma(W_3 \mathbf{z}_2 + \mathbf{b}_3),\quad \boxed{\mathbf{x}^{(l+1)} = \alpha^{(l)} \mathbf{h}^{(l)} + (1-\alpha^{(l)}) \mathbf{x}^{(l)}} $$
with `alpha^(l) in R` trainable, initialised to **0**. `U, V = sigma(W_{1,2} Phi(x) + b_{1,2})` gate every block (modified MLP). Coordinate embedding is Random Fourier Features `Phi(x) = [cos(Bx); sin(Bx)]` with `B ~ N(0, s^2)`.

**Physics-informed initialisation**: with all `alpha^(l)=0` the network reduces to `u_theta(x) = W^(L+1) Phi(x)`. Initialise the last linear layer by `W = argmin || W Phi - Y ||^2` where `Y` is data (IC samples for time-dependent PDEs, or solution of the linearised PDE).

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class RFF(nn.Module):
    m: int = 128
    s: float = 1.0
    @nn.compact
    def __call__(self, x):
        B = self.variable("buffers", "B",
                          lambda: jax.random.normal(self.make_rng("buf"), (self.m, x.shape[-1])) * self.s).value
        proj = x @ B.T
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)

class PirateBlock(nn.Module):
    H: int = 256
    @nn.compact
    def __call__(self, x, U, V):
        f  = nn.tanh(nn.Dense(self.H)(x))
        z1 = f * U + (1 - f) * V
        g  = nn.tanh(nn.Dense(self.H)(z1))
        z2 = g * U + (1 - g) * V
        h  = nn.tanh(nn.Dense(self.H)(z2))
        alpha = self.param("alpha", nn.initializers.zeros, ())   # alpha = 0 at init
        return alpha * h + (1 - alpha) * x

class PirateNet(nn.Module):
    m: int = 128
    H: int = 256
    L: int = 3
    s: float = 1.0
    d_out: int = 1
    @nn.compact
    def __call__(self, x):
        phi = RFF(m=self.m, s=self.s)(x)
        U = nn.tanh(nn.Dense(self.H, name="U_enc")(phi))
        V = nn.tanh(nn.Dense(self.H, name="V_enc")(phi))
        h = nn.Dense(self.H, name="proj")(phi)
        for i in range(self.L):
            h = PirateBlock(H=self.H, name=f"blk{i}")(h, U, V)
        return nn.Dense(self.d_out, use_bias=False, name="out")(h)

def physics_init(params, x_data, y_data):
    # forward up through projection with alpha=0 (identity)
    phi = RFF(m=128).apply({"params": params["params"]["RFF_0"]},
                           x_data, rngs={"buf": jax.random.PRNGKey(0)})
    h = phi @ params["params"]["proj"]["kernel"] + params["params"]["proj"]["bias"]
    W, *_ = jnp.linalg.lstsq(h, y_data)              # solve min ||h W - y||
    params["params"]["out"]["kernel"] = W
    return params

# Loss = causal-weighted (IC + PDE residual + BC) with RFF+modified-MLP backbone
optimizer = optax.chain(
    optax.adam(1e-3),
    optax.scale_by_schedule(optax.exponential_decay(1.0, 2000, 0.9)),
)
# warm-up 5k steps lr 0 -> 1e-3, then exp decay 0.9 per 2k steps, +NTK reweighting
```

Hyperparameters: 3-9 residual blocks (depth 9-27); width `H=256`; Gaussian RFF embedding, scale `s` per PDE (1.0–15.0); Tanh; Random Weight Factorisation; Adam with 5k linear warm-up to `lr=1e-3` then exponential decay 0.9 every 2k–10k steps (per PDE); learning-rate annealing + causal training. **Critical**: `alpha=0` initialisation; LS-init of the output layer to fit IC or linearised PDE solution.

## Results
SOTA on Allen-Cahn (rel-L2 `2.24e-5` vs `5.37e-5` for JAX-PI), KdV (`4.27e-4`), Grey-Scott (`3.61e-3`), Ginzburg-Landau (`1.49e-2`), lid-driven cavity Re=3200 (`4.21e-2`). Unlike plain MLP / ResNet (whose error grows with depth, up to ~100% at 18 layers), PirateNet error keeps decreasing with depth, confirming the adaptive skip restores trainability.
