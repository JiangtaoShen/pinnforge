---
slot: 99
title: "A flow-aware training strategy for physics-informed neural networks"
authors: [Pietro Cestola, Luciano Teresi, Antonio De Simone]
year: 2026
venue: Computer Methods in Applied Mechanics and Engineering 455 (118910)
gitrepo: "https://github.com/pcestola/Flow-Aware-PINN-Training"
---

## TL;DR
Mask the PDE-residual loss to only the collocation points whose geodesic distance to the BC/IC "information boundary" is below an epoch-dependent threshold, and grow that threshold on an exponential schedule. Early epochs supervise only points already constrained by data; later epochs add more distant points as the physics has had time to propagate. No architectural changes, same loss, only the sampling mask evolves.

## Problem
Even with uniform collocation sampling, PINN error empirically decreases first near BC/IC and then propagates inward (Fig. 1 of the paper). Early-epoch residual gradients far from the information boundary correlate poorly with the true error gradient -- they can be useless or actively harmful. Training the full domain from epoch 0 wastes capacity and computation.

## Method
Mesh `M = (V, E, F)` discretizes the space-time domain `D`. Information boundary `Gamma subset partial D` is where IC/BC data lives.

1) Geodesic decomposition. For each vertex compute the multi-source Dijkstra distance to `V cap Gamma`, normalize to `[0, 1]`, then partition the `n` sub-bands
$$
D_i = \{\sigma\in F: \min_{v\in\sigma}\phi(v)\in[(i-1)/n,\, i/n)\},\quad
\mathcal D_k = \bigcup_{i=1}^{k} D_i \cup \Gamma
$$
yielding nested `Gamma subset D_1 subset D_2 subset ... subset D_n = D`. Cost: `O((|V|+|E|) log|V| + |C| log|V|)` once.

2) Exponential epoch schedule (best of the three studied; linear and proportional also implemented):
$$
e_0=0,\quad e_i = e_{i-1} + \Big\lfloor \frac{\exp(i/n)}{\sum_{j=1}^n \exp(j/n)}\,e_{\text{tot}} \Big\rfloor
$$

3) Flow-aware update: for `k in [e_{i-1}, e_i)` train only on points in `C cap D_i`:
$$
\theta_{k+1} = \theta_k - \eta_k\,\mathcal O\!\big[\nabla_\theta \mathcal L(\theta_k; C\cap \mathcal D_i)\big]
$$
where the loss is the standard
$$
\mathcal L(\theta; C) = \mathcal L_\mathcal F(\theta; C\cap D) + \sum_j \lambda_j \mathcal L_{B_j}(\theta; C\cap \Gamma_j)
$$

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import networkx as nx, numpy as np
from scipy.spatial import cKDTree

class MLP(nn.Module):
    hidden: int = 32; depth: int = 3
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.tanh(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x)

def geodesic_decomp(mesh_V, mesh_E, Gamma_idx, collocation, n_bands):
    G = nx.Graph()
    for (i, j) in mesh_E:
        G.add_edge(int(i), int(j), weight=float(np.linalg.norm(mesh_V[i] - mesh_V[j])))
    dist = nx.multi_source_dijkstra_path_length(G, sources=set(Gamma_idx.tolist()))
    d_v   = np.array([dist[v] for v in range(len(mesh_V))])
    phi_v = d_v / d_v.max()
    nn_   = cKDTree(mesh_V).query(collocation)[1]
    phi_c = phi_v[nn_]
    return [np.where((phi_c >= (i-1)/n_bands) & (phi_c < i/n_bands))[0]
            for i in range(1, n_bands + 1)]

def exp_schedule(n, e_tot):
    w = np.exp(np.arange(1, n+1) / n); w = w / w.sum()
    e = np.cumsum((w * e_tot).astype(int))
    return np.concatenate([[0], e])

bands  = geodesic_decomp(V, E, Gamma_idx, np.asarray(X_col), n_bands=32)
e_arr  = exp_schedule(32, e_tot=10_000)
active = bands[0]

net = MLP()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
sched  = optax.linear_schedule(init_value=1e-2, end_value=1e-4, transition_steps=10_000)
optimizer = optax.adam(sched)
opt_state = optimizer.init(params)

def total_loss(params, X_act, X_bcs, lams):
    L = pde_residual_loss(params, X_act)
    for j, X_bc in enumerate(X_bcs):
        L = L + lams[j] * bc_loss_j(params, X_bc, j)
    return L

@jax.jit
def step(params, opt_state, X_act, X_bcs, lams):
    g = jax.grad(total_loss)(params, X_act, X_bcs, lams)
    updates, opt_state = optimizer.update(g, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

cur_band = 0
for k in range(10_000):
    while cur_band + 1 < len(e_arr) and k >= e_arr[cur_band + 1]:
        cur_band += 1
        active = np.concatenate(bands[:cur_band + 1])
    X_act = X_col[active]
    params, opt_state = step(params, opt_state, X_act, X_bcs, lams)
```

Hyperparameters: 3 hidden layers x 32 neurons, tanh, Adam (b1=0.9, b2=0.999), lr 1e-2 -> 1e-4 linear over 10k epochs, n=32 bands, exponential schedule. The information boundary is problem-specific (e.g. `[-1,1]x{0}` for Burgers/wave; the whole partial-Omega for Laplace and lid-driven cavity).

## Results
Seven PINNacle 2-D / 1-D-time benchmarks (1-D Burgers, 1-D wave, 2-D Laplace with holes, lid-driven cavity Re=100, backward-facing step, Eikonal, plus parameterized geometry). With 32 sub-bands the method always matches or beats baseline accuracy while cutting compute by 9-37% GFLOPs (Burgers -10.1%, wave -37.2%, Laplace -9.2%, lid-cavity -17.8%, backward step -15.8%, Eikonal -15.9%). No architectural changes; pure sampling-mask schedule.
