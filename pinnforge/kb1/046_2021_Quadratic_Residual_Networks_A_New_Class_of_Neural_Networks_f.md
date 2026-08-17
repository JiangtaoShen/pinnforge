---
slot: 046
title: "Quadratic Residual Networks: A New Class of Neural Networks for Solving Forward and Inverse Problems in Physics Involving PDEs"
authors: [Jie Bu, A. Karpatne]
year: 2021
venue: SIAM SDM 2021 (arXiv:2101.08366)
gitrepo: "https://github.com/jayroxis/qres"
doi: 10.1137/1.9781611976700.76
---

## TL;DR
QRes adds a Hadamard-product "quadratic residual" term to every layer: `y = σ(W₂x ⊙ W₁x + W₁x + b)`. The extra multiplicative branch doubles the polynomial degree expressible per layer, giving exponentially better width-efficiency than plain MLPs and faster convergence on PINN forward/inverse problems with comparable or fewer parameters.

## Problem
PINNs need expressive networks to fit high-frequency PDE solutions, but plain MLPs of equal capacity are wide/deep and slow to train. Existing quadratic networks (QDN) lack theoretical justification and weren't designed for PINN losses.

## Method
Replace each MLP layer
$$ y_{\text{DNN}} = \sigma(Wx + b) $$
with a quadratic residual layer
$$ y_{\text{QRes}} = \sigma\!\big(\underbrace{W_2 x \odot W_1 x}_{\text{quadratic residual}} + W_1 x + b\big) $$
where `⊙` is Hadamard product, `W_1, W_2 ∈ ℝ^{d_out × d_in}`, `b ∈ ℝ^{d_out}`. Setting `W_2 = 0` recovers a plain DNN; otherwise each layer can represent a degree-2 polynomial of its inputs in addition to the linear sum. Stacking `h` layers raises the polynomial degree to `(2r)^{h-1}` (for activation with leading degree `r`), versus `r^{h-1}` for plain MLPs. With `r=1` (linear) a deep QRes already represents polynomials of degree `2^{h-1}`. Use `tanh` activation (bounded) to prevent blow-up.

Theoretical guarantees (Theorems 4.1 & 4.2 of the paper):
- *Depth efficiency*: an MLP needs `h_n ≥ 1 + (h_q-1)(1 + log 2 / log r)` layers to match a filling QRes of depth `h_q`.
- *Width efficiency*: at equal expressive power, MLP width grows as `O(2^τ)` faster than QRes width.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class QResLayer(nn.Module):
    d_out: int

    @nn.compact
    def __call__(self, x):
        w1x = nn.Dense(self.d_out, use_bias=True,  name="W1")(x)   # W1 x + b
        w2x = nn.Dense(self.d_out, use_bias=False, name="W2")(x)   # W2 x
        return jnp.tanh(w2x * w1x + w1x)                           # element-wise ⊙

class QRes(nn.Module):
    hidden: int
    out_dim: int
    depth:  int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = QResLayer(self.hidden)(x)
        return nn.Dense(self.out_dim)(x)                            # linear final layer

# drop into a PINN
model = QRes(hidden=20, out_dim=1, depth=4)

def pinn_loss(params, x_b, u_b, x_f):
    def u_of(xx): return model.apply(params, xx)
    u    = u_of(x_f)
    u_x  = jax.grad(lambda xx: u_of(xx).sum())(x_f)
    res  = pde_operator(u, u_x, x_f)
    return jnp.mean(res ** 2) + jnp.mean((u_of(x_b) - u_b) ** 2)

params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
opt    = optax.adam(1e-3)
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, x_b, u_b, x_f):
    g = jax.grad(pinn_loss)(params, x_b, u_b, x_f)
    upd, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state

for _ in range(5_000):
    params, opt_state = step(params, opt_state, x_b, u_b, x_f)

# refine with L-BFGS (jaxopt)
from jaxopt import LBFGS
solver = LBFGS(fun=lambda p: pinn_loss(p, x_b, u_b, x_f),
               maxiter=50, history_size=50, tol=1e-9)
params, _ = solver.run(params)
```

Recommended hyperparameters (paper Table 1/4): hidden=20, depth=4–8 for QRes (≈√2× narrower than the matched PINN), tanh activation, Adam → L-BFGS schedule, same training points/epochs as the baseline PINN.

## Results
On forward Burgers, Schrödinger, Allen-Cahn and inverse Navier-Stokes, Burgers and KdV problems, QRes matches or beats PINN accuracy with 0.5–1× the parameter count and similar or fewer Adam epochs. Width sweep on Burgers (28 architectures): QRes is uniformly more parameter-efficient. The quadratic term is especially helpful for high-frequency solutions.
