---
slot: 038
title: "Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations"
authors: [Ben Moseley, Andrew Markham, Tarje Nissen-Meyer]
year: 2021
venue: Advances in Computational Mathematics 49:62 (2023)
gitrepo: "https://github.com/benmoseley/FBPINNs"
doi: 10.1007/s10444-023-10065-9
---

## TL;DR
FBPINNs partition the domain into overlapping subdomains, place a small MLP in each, and form the global solution as a sum of subdomain networks each multiplied by a smooth compactly-supported window — like a learned FEM basis. With per-subdomain input normalisation and flexible "active/fixed/inactive" training schedules, FBPINNs solve large multi-scale PDEs that defeat vanilla PINNs.

## Problem
PINNs suffer spectral bias and ill-conditioned losses as the domain grows: the effective frequency seen by the network scales with domain size, so vanilla PINNs fail on `du/dx = cos(ωx)` for `ω ≥ 15`. XPINN-style decompositions need interface penalty terms that introduce discontinuities.

## Method
The solution ansatz is a sum of windowed subdomain networks with a hard-constraint operator `C`:
$$ \hat{u}(x;\theta) = C\Big[\sum_{i=1}^{n} w_i(x)\,\text{unnorm}\circ NN_i\circ \text{norm}_i(x)\Big] $$
Window function (hyperrectangular subdomain `Ω_i`):
$$ w_i(x) = \prod_{j=1}^{d} \phi\!\left(\frac{x_j-a_{ij}}{\sigma_{ij}}\right)\phi\!\left(\frac{b_{ij}-x_j}{\sigma_{ij}}\right) $$
with `φ` the sigmoid. Because `w_i` vanishes outside `Ω_i` and the sum is smooth, continuity across interfaces is automatic — no interface loss is needed. Only the physics loss is minimised:
$$ \mathcal{L}(\theta) = \frac{1}{N_p}\sum_i \|\mathcal{D}[\hat{u}(x_i;\theta);\lambda] - f(x_i)\|^2 $$
Each subdomain has its own input normalisation to `[-1,1]`, which keeps the effective frequency low. Training schedules mark each network as active (updated), fixed (frozen), or inactive (zero output), enabling time-marching style sweeps outward from the boundary.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class SubNet(nn.Module):
    hidden: int = 16
    depth: int  = 2
    d_out: int  = 1
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(self.d_out)(x)

def window(x, a, b, sigma):
    # x: (N,d), a,b,sigma: (d,)
    s = jax.nn.sigmoid((x - a) / sigma) * jax.nn.sigmoid((b - x) / sigma)
    return jnp.prod(s, axis=-1, keepdims=True)

def fbpinn_forward(params_list, model, x, centers, halfwidths, sigmas, u_scale, hard_bc):
    u_sum = jnp.zeros((x.shape[0], 1))
    for i, p in enumerate(params_list):
        a, b = centers[i] - halfwidths[i], centers[i] + halfwidths[i]
        w  = window(x, a, b, sigmas[i])
        xn = 2 * (x - centers[i]) / (2 * halfwidths[i])      # norm to [-1,1]
        ui = model.apply(p, xn) * u_scale
        u_sum = u_sum + w * ui
    return hard_bc(x, u_sum)                                  # e.g. tanh(ωx)*NN

model = SubNet()
params_list = [model.init(jax.random.PRNGKey(i), jnp.zeros((1, d_in)))
               for i in range(n_subdomains)]
opt = optax.adam(1e-3)
opt_states = [opt.init(p) for p in params_list]

def loss_fn(p_i, i, params_list, x):
    pl = list(params_list); pl[i] = p_i
    u  = fbpinn_forward(pl, model, x, centers, halfwidths, sigmas, u_scale, hard_bc)
    res = pde_residual(u, x)
    return jnp.mean(res ** 2)

@jax.jit
def update_subdomain(i, params_list, opt_states, x):
    grads = jax.grad(loss_fn)(params_list[i], i, params_list, x)
    updates, opt_states_i = opt.update(grads, opt_states[i], params_list[i])
    new_params = optax.apply_updates(params_list[i], updates)
    return new_params, opt_states_i

# training: only active subdomains carry updates
for step in range(50_000):
    for i in active_models:
        x = sample_in_subdomain(i)
        params_list[i], opt_states[i] = update_subdomain(i, params_list, opt_states, x)
```
Defaults: tanh MLPs with 2 hidden layers × 16 units per subdomain; Adam lr=1e-3; overlap width comparable to subdomain width; output unnormalised by `1/ω`; same training points reused across overlapping nets in step 2.

## Results
On `du/dx=cos(15x)` over `[-2π,2π]`, FBPINN with 15 subdomains converges to L1 ≈ 1e-3 while a comparable single PINN (321 params) fails and a 66 k-param PINN gives much worse accuracy after 5× the compute. FBPINNs also match PINN accuracy on Burgers (1+1D) and robustly solve a (2+1)D wave problem where the PINN diverges, using a time-marching schedule.
