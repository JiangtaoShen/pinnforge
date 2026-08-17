---
slot: 16
title: "Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks"
authors: [Ameya D. Jagtap, Kenji Kawaguchi, George Em Karniadakis]
year: 2020
venue: "Proceedings of the Royal Society A"
gitrepo: "https://github.com/AmeyaJagtap/Locally-Adaptive-Activation-Functions-Neural-Networks-"
doi: "10.1098/rspa.2020.0334"
---

## TL;DR
Extend Jagtap-Karniadakis (slot 001) global-slope adaptive activation to *layer-wise* (L-LAAF) and *neuron-wise* (N-LAAF) trainable slopes. Add a *slope-recovery* regulariser `S(a) = 1 / [(1/(D-1)) sum_k exp(a_k)]` to the loss to push slopes to large values automatically. Faster convergence and lower final error on PINN benchmarks and on standard image-classification benchmarks (CIFAR, MNIST, ...).

## Problem
A single global slope `a` (GAAF) gives one degree of freedom only; different layers/neurons need different slopes for optimal training. Vanishing gradients in deep PINNs stall convergence; a mechanism to maintain non-zero slopes is needed.

## Method

### A. L-LAAF (one slope per hidden layer)
For each hidden layer `k = 1..D-1`, introduce trainable `a_k in R` and apply:
$$
\sigma(n\,a_k\,L_k(z_{k-1}))
$$
with fixed scaling factor `n >= 1` (paper uses `n=5` for PINNs). Initialise `n a_k = 1`.

### B. N-LAAF (one slope per neuron)
For each neuron `i = 1..N_k` in layer `k`:
$$
\sigma\bigl(n\,a^{(i)}_k\,[L_k(z_{k-1})]_i\bigr)
$$
adds `sum_k N_k` parameters; for typical depth/width the parameter growth is <10%.

### C. Slope-recovery term
Augment the loss to drive slopes large (helps gradients flow):
$$
S(a) = \frac{1}{\dfrac{1}{D-1}\sum_{k=1}^{D-1}\exp(a_k)}
$$
(layer-wise; replace `a_k` by `(1/N_k) sum_i a_k^{(i)}` for N-LAAF). Total PINN loss:
$$
J(\hat\Theta) = W_F\,\mathrm{MSE}_F + W_u\,\mathrm{MSE}_u + W_a\,S(a)
$$

Theoretical properties (paper proofs):
- Under mild conditions on `lr` and init, gradient descent on this loss avoids sub-optimal critical points.
- Gradient dynamics of LAAF is equivalent to a *preconditioned* gradient flow of the base model — implicit conditioning matrix that no constant learning-rate schedule can reproduce.

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class LAAF_MLP(nn.Module):
    h: int
    depth: int
    out_dim: int
    n: float = 5.0
    mode: str = "layer"        # "layer" or "neuron"

    @nn.compact
    def __call__(self, x):
        if self.mode == "layer":
            a = self.param("a", lambda key: jnp.full((self.depth,), 1.0 / self.n))
        else:  # neuron-wise: one slope per neuron per layer
            a = self.param("a", lambda key: jnp.full((self.depth, self.h), 1.0 / self.n))
        for k in range(self.depth):
            z = nn.Dense(self.h)(x)
            scale = a[k] if self.mode == "layer" else a[k][None, :]
            x = jnp.tanh(self.n * scale * z)
        return nn.Dense(self.out_dim)(x)

def slope_recovery(params, mode):
    a = params["params"]["a"]
    if mode == "layer":
        return 1.0 / jnp.mean(jnp.exp(a))
    return 1.0 / jnp.mean(jnp.exp(a.mean(axis=1)))     # average per-layer

net = LAAF_MLP(h=20, depth=5, out_dim=1, n=5.0, mode="neuron")
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

W_F, W_u, W_a = 1.0, 1.0, 1.0

def loss_fn(params, xt_r, xt_u, u_target):
    L_F = jnp.mean(pde_residual(net, params, xt_r) ** 2)
    L_u = jnp.mean((net.apply(params, xt_u) - u_target) ** 2)
    L_a = slope_recovery(params, net.mode)
    return W_F * L_F + W_u * L_u + W_a * L_a

@jax.jit
def train_step(params, opt_state, xt_r, xt_u, u_target):
    grads = jax.grad(loss_fn)(params, xt_r, xt_u, u_target)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

Recommended: `n=5` (PINN), `n=1-2` (image classification); init `n a_k = 1`; `W_a = 1`; Adam `lr=1e-3`; depth 4-8.

## Results
On Helmholtz, Burgers, Klein-Gordon inverse problems, L-LAAF + slope recovery converges ~2-5x faster than GAAF and reaches ~10x lower relative L2 error. N-LAAF is slightly better than L-LAAF but with marginally higher cost. On CIFAR-10/100, SVHN, MNIST, KMNIST etc. test accuracy improves by 0.5-2 percentage points over fixed-activation baselines.
