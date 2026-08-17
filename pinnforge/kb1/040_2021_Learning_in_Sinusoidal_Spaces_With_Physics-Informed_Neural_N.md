---
slot: 040
title: "Learning in Sinusoidal Spaces With Physics-Informed Neural Networks"
authors: [Jian Cheng Wong, Chinchun Ooi, Abhishek Gupta, Yew-Soon Ong]
year: 2021
venue: IEEE Transactions on Artificial Intelligence
gitrepo: ""
---

## TL;DR
A standard Xavier-initialised tanh PINN has near-zero input gradients at initialisation, so it trivially satisfies the PDE residual but is far from any BC-compatible solution — a deceptive local minimum SGD cannot escape. Prepending a learnable sinusoidal feature map `γ(x)=sin(2π(W₁x+b₁))` with `W₁~N(0,σ²)` boosts the input-gradient variance everywhere, yielding "sf-PINN" architectures that train orders of magnitude better across forward and inverse PDE problems.

## Problem
Authors prove (Proposition 1) that as width `n→∞`, an MLP-tanh PINN with Xavier init has `Var(∂u/∂x) ∝ 1/n → 0`, so `u(x;w)≈const` and most physical PDEs (heat, wave, Laplace, NS) are trivially satisfied at init. SGD cannot move because both the PDE residual and `∂u/∂x` are ~0. Standard remedies (loss balancing, He init) only partially help.

## Method
Replace the first layer of the PINN with a sinusoidal feature map and train the rest as usual:
$$ \gamma(x) = \sin\!\big(2\pi(W_1 x + b_1)\big),\quad W_1\in\mathbb{R}^{n_1\times d},\ b_1\in\mathbb{R}^{n_1} $$
$$ u(x;w) = \text{MLP}_{\tanh}\!\big(\gamma(x)\big) $$
Initialise `W_1 ~ N(0, σ²)`, `b_1 = 0`. Proposition 3 shows the input-gradient variance becomes
$$ \mathrm{Var}\!\left(\tfrac{\partial u}{\partial x}\right) \le 2\pi^2 \sigma^2 \big(1 - e^{-16\pi^2\sigma^2 x^2}\big) $$
i.e. ≈ `4π²σ²` at `x=0` and a near-constant upper bound `2π²σ²` for `|x|≫0`. Thus `σ` (the "bandwidth") directly modulates the initial frequency content of the network and should match the dominant frequency of the target solution.

Variants explored: sf-PINN (sin first layer + tanh hidden), SIREN (sin everywhere + He init), ff-PINN (sin & cos pairs sharing `W_1`), rf-PINN (sin & cos pairs with `W_1` frozen). All behave similarly when `σ` is tuned.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax, math

class SfPINN(nn.Module):
    d_in:   int
    n1:     int = 64
    hidden: int = 50
    depth:  int = 3
    d_out:  int = 1
    sigma:  float = 1.0

    @nn.compact
    def __call__(self, x):
        W1 = self.param('W1', nn.initializers.normal(stddev=self.sigma),
                        (self.n1, self.d_in))
        b1 = self.param('b1', nn.initializers.zeros, (self.n1,))
        feat = jnp.sin(2 * math.pi * (x @ W1.T + b1))
        h = feat
        for _ in range(self.depth):
            h = jnp.tanh(nn.Dense(self.hidden,
                                  kernel_init=nn.initializers.xavier_normal())(h))
        return nn.Dense(self.d_out)(h)

model = SfPINN(d_in=d_in)

# typical loss (e.g. convection-diffusion)
def pinn_loss(params, x_f, x_b, u_b, lam=500.0):
    def u_of(xx): return model.apply(params, xx)
    u    = u_of(x_f)
    u_x  = jax.grad(lambda xx: u_of(xx).sum())(x_f)
    u_xx = jax.grad(lambda xx: jax.grad(lambda yy: u_of(yy).sum())(xx).sum())(x_f)
    res  = v * u_x - k * u_xx
    return lam * jnp.mean(res ** 2) + jnp.mean((u_of(x_b) - u_b) ** 2)

params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, d_in)))
schedule = optax.exponential_decay(5e-3, 5_000, 0.5, end_value=1e-6)
opt = optax.adam(schedule)
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, x_f, x_b, u_b):
    grads = jax.grad(pinn_loss)(params, x_f, x_b, u_b)
    upd, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, upd), opt_state

for it in range(50_000):
    params, opt_state = step(params, opt_state, x_f, x_b, u_b)
```

Recommended hyperparameters (Table I/III):
- `n1 = 64` sinusoidal features, 3–4 tanh hidden layers × 50 units.
- `σ ∈ [0.5, 2.5]`; grid-search in `[1e-1, 1e1]` when the solution frequency is unknown.
- `λ_PDE` weight from 1–500; sf-PINN is much less sensitive to it than baseline.
- Adam lr=5e-3, exponential decay down to 1e-6, 50k–200k iters.

## Results
On 1-D steady convection–diffusion, 1-D wave, 2-D Taylor-Green NS, 1-D KdV, 2-D Helmholtz and lid-driven cavity at Re=400, sf-PINN reduces MSE by 1–4 orders of magnitude over Xavier-tanh PINN and is robust across activations (tanh/sin/sigmoid) and inits (Xavier/He). SIREN, ff-PINN, rf-PINN are essentially equivalent once `σ` is tuned, supporting the unified "learning in sinusoidal spaces" view.
