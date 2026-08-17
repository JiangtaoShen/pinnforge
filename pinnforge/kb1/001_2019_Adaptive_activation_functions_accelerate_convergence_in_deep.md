---
slot: 1
title: "Adaptive activation functions accelerate convergence in deep and physics-informed neural networks"
authors: [Ameya Dilip Jagtap, G. Karniadakis]
year: 2019
venue: "Journal of Computational Physics (arXiv:1906.01170)"
gitrepo: ""
---

## TL;DR
Introduce a single globally-shared scalar `a` inside every activation `sigma(n*a*Lk(x))` together with a fixed scaling factor `n>=1`. The slope `a` is trained jointly with weights/biases via the same Adam step, which dramatically accelerates convergence of PINNs in the early phase and improves final accuracy on Klein-Gordon, Burgers and Helmholtz.

## Problem
Vanilla PINNs use a fixed activation (tanh/sigmoid/ReLU); the loss-landscape topology and the spectral bias depend strongly on activation slope, leaving early-training convergence very slow. Tuning slope by hand requires trial-and-error.

## Method
Replace each pre-activation `Lk(x_{k-1}) = W_k x_{k-1} + b_k` by `sigma(n * a * Lk(x_{k-1}))`, where:
- `a` is a single scalar trainable parameter (one per whole network),
- `n >= 1` is a fixed scaling factor that multiplies the effective learning rate of `a` so its update keeps pace with the weights.

Optimisation problem (Adam over `Theta_tilde = {W_k, b_k, a}`):
$$
\Theta^{*} = \arg\min_{\Theta,\,a\in\mathbb{R}_{>0}} \; J(\Theta,a), \quad J = \mathrm{MSE}_F + \mathrm{MSE}_u
$$
$$
a^{m+1} = a^{m} - \eta_l \,\nabla_a J^{m}(a), \qquad u_\Theta(x) = (L_D \circ \sigma_a \circ L_{D-1} \circ \cdots \circ \sigma_a \circ L_1)(x)
$$
where `sigma_a(z) = sigma(n * a * z)`.

Initialisation: `a_0 = 1/n` so that `n * a_0 = 1` at start (recovers standard activation). Larger `n` (e.g. 5, 10) accelerates training but too-large `n` destabilises optimisation.

JAX (flax.linen + optax):
```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class AdaptiveMLP(nn.Module):
    hidden: int
    out_dim: int
    depth: int
    n: float = 5.0
    act: callable = jnp.tanh

    @nn.compact
    def __call__(self, x):
        # ONE global scalar slope shared across all hidden layers.
        a = self.param("a", lambda key: jnp.array(1.0 / self.n))
        for i in range(self.depth):
            x = nn.Dense(self.hidden, name=f"L{i}")(x)
            x = self.act(self.n * a * x)
        return nn.Dense(self.out_dim, name="Lout")(x)

net = AdaptiveMLP(hidden=20, out_dim=1, depth=6, n=5.0)  # Burgers config
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss
# `a` lives in `params["params"]["a"]` and is updated jointly with weights.
```

Recommended hyperparameters (from paper):
- Klein-Gordon: depth 2, width 40, `n in {1..10}`
- Burgers: depth 6, width 20, `n in {1, 5}`
- Helmholtz: depth 4, width 20
- Optimiser: Adam, `lr = 1e-3` (followed by L-BFGS in some runs)
- `a` initialised at `1/n`

Theoretical note: paper proves Adam with this parameterisation cannot be trapped at sub-optimal critical points under mild assumptions on init/lr.

## Results
On Burgers (nu = 0.01/pi), adaptive activation with n=5 reduces relative L2 from O(1e-2) (fixed activation) to ~1e-3. Klein-Gordon and Helmholtz forward problems converge ~2-5x faster in epochs. Inverse problems (parameter identification) achieve <1% parameter error with the same epoch budget.
