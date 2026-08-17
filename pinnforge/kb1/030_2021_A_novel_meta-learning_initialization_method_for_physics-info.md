---
slot: 030
title: "A novel meta-learning initialization method for physics-informed neural networks"
authors: [Xu Liu, Xiaoya Zhang, Wei Peng, Weien Zhou, Wen Yao]
year: 2021
venue: "Neural Computing and Applications (arXiv:2107.10991)"
gitrepo: ""
---

## TL;DR
NRPINN: Apply the Reptile meta-learning algorithm to a family of parameterized PDEs (e.g. Burgers at different viscosities, Poisson with different sources). Meta-trained initial weights make a NEW PINN converge much faster and to lower error than Xavier init. Unsupervised, supervised, and semi-supervised variants are supported.

## Problem
Vanilla PINN training from Xavier init takes 10^4-10^5 Adam steps per problem. For many-query settings (parametric studies, inverse problems, control) this is prohibitive. Prior accelerations (adaptive activations, Fourier features, decomposition) ignore initialization quality.

## Method
Treat each value of the PDE parameter lambda (or each labeled dataset) as a separate task tau. Meta-objective: find phi such that K steps of inner Adam on any task tau gives a near-optimal model.

Reptile outer loop (Algorithm 2):
$$
\phi \leftarrow \phi + \epsilon(\tilde\phi - \phi),\quad \tilde\phi = U^k_{\tau}(\phi)
$$
where U_tau^k denotes k inner SGD/Adam steps on task tau using the appropriate loss:
- Supervised (zero-order): L_z = MSE between u_phi and labeled samples u_lambda.
- Unsupervised (high-order): L_h = PDE residual loss + BC loss for that lambda.
- Semi-supervised: alternate.

Outer step size epsilon = epsilon_0 (1 - j/N) (linear decay). After meta-training, take phi^* as initialization and train a fresh PINN on the target task.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax, random

class MLP(nn.Module):
    width: int = 40; depth: int = 3
    @nn.compact
    def __call__(self, xt):
        for _ in range(self.depth):
            xt = nn.tanh(nn.Dense(self.width)(xt))
        return nn.Dense(1)(xt)

net = MLP()
apply_fn = net.apply

def burgers_residual_point(params, xt, lam):
    def u_fn(z): return apply_fn(params, z)[0]
    g = jax.grad(u_fn)(xt); H = jax.hessian(u_fn)(xt)
    return g[1] + u_fn(xt) * g[0] - lam * H[0,0]

def loss_for_lambda(params, lam, xt_r, xt_b, x_ic, u_ic):
    r = jax.vmap(lambda z: burgers_residual_point(params, z, lam))(xt_r)
    L_r  = jnp.mean(r**2)
    L_b  = jnp.mean(jax.vmap(lambda z: apply_fn(params, z)[0])(xt_b)**2)
    L_ic = jnp.mean((jax.vmap(lambda z: apply_fn(params, z)[0])(x_ic) - u_ic)**2)
    return L_r + L_b + L_ic

def inner_update(params, lam, batch, k=5, lr=1e-3):
    opt = optax.adam(lr); state = opt.init(params)
    @jax.jit
    def istep(params, state):
        g = jax.grad(loss_for_lambda)(params, lam, *batch)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state
    for _ in range(k):
        params, state = istep(params, state)
    return params

# ----- Meta-training (Reptile) -----
phi = net.init(jax.random.PRNGKey(0), jnp.zeros(2))
tasks = [lam_min + i*(lam_max-lam_min)/N_TASKS for i in range(N_TASKS)]
N_OUTER, K_INNER, eps0 = 5000, 5, 0.1

for j in range(N_OUTER):
    lam = random.choice(tasks)
    batch = (xt_r, xt_b, x_ic, u_ic_for(lam))
    phi_tilde = inner_update(phi, lam, batch, k=K_INNER, lr=1e-3)
    eps = eps0 * (1 - j/N_OUTER)
    # phi <- phi + eps * (phi_tilde - phi)
    phi = jax.tree_util.tree_map(lambda a, b: a + eps*(b - a), phi, phi_tilde)

# ----- Solve target task using phi as init -----
params = jax.tree_util.tree_map(lambda x: x, phi)
opt = optax.adam(1e-3); state = opt.init(params)

@jax.jit
def fine_step(params, state, batch):
    g = jax.grad(loss_for_lambda)(params, lam_target, *batch)
    upd, state = opt.update(g, state, params)
    return optax.apply_updates(params, upd), state
```

Recommended: 3-4 hidden layers x 40 tanh, K=5 inner steps, eps_0=0.1 linearly decayed, N=5000 outer iters, tasks parameterized by viscosity, source frequency, etc.

## Results
On Burgers (varying viscosity), Poisson, and Schrodinger, NRPINN reaches the same L2 error as Xavier-init PINN in ~5-10x fewer Adam steps and ends with ~2-5x lower final error. Works in both forward and inverse modes; integrates with adaptive-activation PINN.
