---
slot: 054
title: "Augmented Physics-Informed Neural Networks (APINNs): A gating network-based soft domain decomposition methodology"
authors: [Zheyuan Hu, Ameya Dilip Jagtap, G. Karniadakis, Kenji Kawaguchi]
year: 2022
venue: Engineering Applications of Artificial Intelligence (arXiv:2211.08939)
gitrepo: ""
---

## TL;DR
APINN replaces XPINN's hard space-time decomposition + interface losses with a *trainable gating network* `G(x): Ω → Δ_m` that softly weights `m` sub-networks `E_i(h(x))` sharing a common trunk `h`. The gate is pre-trained to mimic an XPINN partition and then fine-tuned, so no interface penalties are needed, every sub-net sees all training points, and the trunk `h` shares structure across sub-domains. Gives consistent gains over PINN and XPINN.

## Problem
XPINN partitions Ω into subdomains, trains one sub-PINN per subdomain, and stitches them with interface losses. This (i) splits the training data, hurting each sub-net; (ii) introduces large errors near interfaces because high-order residual continuity is hard to enforce; (iii) the decomposition is fixed a priori.

## Method
**Model.** Shared trunk `h: ℝ^d → ℝ^H`, `m` heads `E_i: ℝ^H → ℝ`, and a gate `G: ℝ^d → Δ^m`:
$$ u_\theta(x) = \sum_{i=1}^{m} G(x)_i\,E_i(h(x)),\qquad \sum_i G(x)_i = 1 $$
- If `G` is trainable → **APINN**; if frozen → **APINN-F**.
- The placement of `h` inside the sum (eq. 10) makes each `E_i∘h` interpretable as the i-th component function.

**Loss.** Plain PINN composite — **no interface losses**:
$$ \mathcal{R}_S(\theta) = \frac{1}{n_b}\sum_i |u_\theta(x_{b,i})-g(x_{b,i})|^2 + \frac{1}{n_r}\sum_i |L u_\theta(x_{r,i}) - f(x_{r,i})|^2 $$
Every collocation point trains every sub-net (unlike XPINN).

**Gate pre-training.** Choose any XPINN-style hard partition `Ω = ∪ Ω_i` and a smooth mimic of its indicators. Examples:
- Upper/lower split at `t=0`: `(G)_1 = e^{t-1}`, `(G)_2 = 1 - e^{t-1}`.
- Inner box vs outer: `(G)_1 = exp(-5(x-0.5)² - 5(t-0.5)²)`, `(G)_2 = 1 - (G)_1`.

Pre-train `G` by regression on these targets, then optionally unfreeze and fine-tune along with everything else.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MLP(nn.Module):
    d_out:  int
    hidden: int
    depth:  int
    last_softmax: bool = False
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.hidden)(x))
        x = nn.Dense(self.d_out)(x)
        return jax.nn.softmax(x, axis=-1) if self.last_softmax else x

class APINN(nn.Module):
    d_in: int
    hidden: int = 40
    depth:  int = 4
    m:      int = 2

    @nn.compact
    def __call__(self, x):
        h     = MLP(d_out=self.hidden, hidden=self.hidden, depth=self.depth,
                    name="trunk")(x)
        outs  = jnp.concatenate([MLP(d_out=1, hidden=self.hidden, depth=2,
                                     name=f"head_{i}")(h) for i in range(self.m)], axis=-1)
        gate  = MLP(d_out=self.m, hidden=self.hidden, depth=self.depth,
                    last_softmax=True, name="gate")(x)
        return jnp.sum(gate * outs, axis=-1, keepdims=True)

model  = APINN(d_in=2, m=2)
params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))

# 1) Pre-train gate to mimic an XPINN partition
def gate_target(x):                                   # upper/lower split at t=0
    t  = x[:, 1:2]
    g0 = jnp.exp(t - 1.0)
    return jnp.concatenate([g0, 1.0 - g0], axis=-1)

def gate_apply(p_gate, x):
    return MLP(d_out=2, hidden=40, depth=4, last_softmax=True).apply(p_gate, x)

opt_g     = optax.adam(1e-3)
state_g   = opt_g.init(params['params']['gate'])

@jax.jit
def gate_step(p_gate, state_g, x):
    def L(pg):
        g = gate_apply(pg, x)
        return jnp.mean(jnp.sum((g - gate_target(x)) ** 2, axis=-1))
    grads = jax.grad(L)(p_gate)
    upd, state_g = opt_g.update(grads, state_g, p_gate)
    return optax.apply_updates(p_gate, upd), state_g

p_gate = params['params']['gate']
for _ in range(2000):
    x = sample_domain(2048)
    p_gate, state_g = gate_step(p_gate, state_g, x)
params['params']['gate'] = p_gate

# 2) Fine-tune everything with plain PINN loss
opt   = optax.adam(1e-3)
state = opt.init(params)

def pinn_loss(params, x_r, x_b, u_b):
    u   = model.apply(params, x_r)
    res = pde_operator(u, x_r)
    return jnp.mean(res ** 2) + jnp.mean((model.apply(params, x_b) - u_b) ** 2)

@jax.jit
def step(params, state, x_r, x_b, u_b):
    g = jax.grad(pinn_loss)(params, x_r, x_b, u_b)
    upd, state = opt.update(g, state, params)
    return optax.apply_updates(params, upd), state

for it in range(50_000):
    params, state = step(params, state, x_r, x_b, u_b)
```

Recommended hyperparameters: tanh MLPs, trunk `h` 4 layers × 40, each head 2 layers × 40, gate 4 layers × 40 → softmax; `m=2–4` heads; Adam lr=1e-3; pre-train gate ~2k steps then unfreeze; standard PINN λ-weighting is enough — *no* interface weight.

**Theory.** Following Hu et al. 2021 generalization bound for XPINN, the paper derives bounds for APINN with trainable and fixed gates showing soft trainable decomposition controls both overfitting (each sub-net sees all data) and target complexity per head.

## Results
Across Helmholtz, Klein-Gordon, Burgers, Allen-Cahn, KdV, and an L-shape Poisson, APINN consistently beats or matches XPINN and PINN. Crucially, the optimised gate visualises the *learned* partition, often differing from the seed and revealing better decompositions; initialising APINN with these gates further improves accuracy, suggesting an automated route to optimal domain decomposition.
