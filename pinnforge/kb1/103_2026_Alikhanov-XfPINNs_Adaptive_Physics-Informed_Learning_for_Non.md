---
slot: 103
title: "Alikhanov-XfPINNs: Adaptive Physics-Informed Learning for Nonlinear Fractional PDEs on Nonuniform Meshes"
authors: [Himanshu Kumar Dwivedi, Matthias J. Ehrhardt, Rajeev]
year: 2026
venue: arXiv:2605.01305
gitrepo: ""
---

## TL;DR
For nonlinear time-fractional PDEs `C_0 d_t^alpha v + N[v;lambda] = 0` (Caputo, alpha in (0,1)), discretize the Caputo derivative with the second-order Alikhanov scheme on a graded mesh `t_n = T (n/K_t)^gamma`, accelerate it with a sum-of-exponentials approximation of the kernel (memory cost O(log K_t) instead of O(K_t)), and plug the resulting residual into a PINN with adaptive `n*tanh(a*x)` activations and hard-imposed IC/BC.

## Problem
Caputo derivatives are non-local so autograd cannot compute them; existing fPINNs use uniform-mesh L1 / power-series schemes which are accuracy-limited and produce huge memory cost. Solutions of time-fractional PDEs have a weak initial singularity `u_t = O(t^{alpha-1})` that uniform meshes don't resolve.

## Method
A. Nonuniform graded mesh `t_n = T (n/K_t)^gamma`, with step ratio `rho = max(tau_k/tau_{k+1}) <= 3/2` (assumption M1) and grading `tau_n <= tau min(1, C1 t_n^{1-1/gamma})`. `gamma >= 1` clusters points near 0.

B. Alikhanov approximation at `t_{k-theta} = theta t_{k-1} + (1-theta) t_k`, `theta = alpha/2`:
$$
(C_0\partial_t^\alpha v)_{k-\theta}\approx \sum_{n=1}^{k} D_{(k-n,k)}\,\nabla_\tau v_n
$$
with kernels `D_{(k-n,k)}` built from `a_{(k-n,k)}` and `b_{(k-n,k)}` integrals of `omega_{1-alpha}` over the graded intervals.

C. Sum-of-exponentials (SOE) speedup: split the convolution into local `[t_{k-1}, t_{k-theta}]` and history `[0, t_{k-1}]`; approximate the kernel `omega_{1-alpha}(t)` by `sum_l nu_l exp(-s_l t)` with `N_q = O(log(1/eps) (log log(1/eps) + log(T/Delta t)) + log(1/Delta t)(log log(1/eps) + log(1/Delta t)))`. The history term then satisfies a recurrence
$$
V^l_{\text{his}}(t_n) = e^{-s_l(\theta\tau_n + (1-\theta)\tau_{n+1})} V^l_{\text{his}}(t_{n-1}) + c_{(n,l)}\nabla_\tau v_n + d_{(n,l)}(\rho_n \nabla_\tau v_{n+1} - \nabla_\tau v_n)
$$
giving cost `O(K_t log K_t)` total.

D. Adaptive activation (Jagtap-Kawaguchi-Karniadakis):
$$
\sigma_a(z) = \tilde\sigma(n a z),\quad a\in\mathbb R\text{ trainable},\;n\text{ fixed scale}
$$
where `tilde sigma` is tanh/sin/Swish/etc. The trainable `a` reshapes gradient flow and accelerates convergence.

E. Hard constraints. Build the ansatz so IC/BC are exact:
$$
v_{\text{NN}}(x,t;\Theta) = \phi_0(x) + t \cdot D(x,t) \cdot \text{MLP}(x,t;\Theta)
$$
where `D(x,t)` is a distance/level-set function vanishing on partial-Omega. Only the PDE-residual loss remains:
$$
\mathcal L(\Theta) = \frac1{N_f}\sum_i \Big|(\,^F\!C_0\partial_t^\alpha v_{NN})_{(x_i,t_i)} + N[v_{NN}(x_i,t_i);\lambda] - g(x_i,t_i)\Big|^2
$$

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class AdaptiveTanh(nn.Module):
    n_scale: float = 2.0
    @nn.compact
    def __call__(self, x):
        a = self.param("a", nn.initializers.ones, ())
        return jnp.tanh(self.n_scale * a * x)

class XfNet(nn.Module):
    in_dim: int = 2; hidden: int = 64; depth: int = 4; n_scale: float = 2.0
    @nn.compact
    def __call__(self, xt):
        h = xt
        for _ in range(self.depth):
            h = AdaptiveTanh(self.n_scale)(nn.Dense(self.hidden)(h))
        return nn.Dense(1)(h)

def hard_ansatz(params, xt, phi0_fn, dist_fn):
    x, t = xt[..., 0:1], xt[..., 1:2]
    return phi0_fn(x) + t * dist_fn(x) * net.apply(params, xt)

def make_graded_mesh(T, K_t, gamma):
    return T * (jnp.arange(K_t + 1).astype(float) / K_t)**gamma

def alikhanov_kernels(t_grid, alpha):
    # Precompute D_{(k-n, k)} for n in 1..k, k in 1..K_t (see eq 12)
    # ... implementation of a_{(k-n,k)}, b_{(k-n,k)} via numerical integration of omega_{1-alpha} ...
    pass

def soe_weights(alpha, T, dt_min, eps=1e-8):
    # Returns nu_l, s_l with sum_l nu_l e^{-s_l t} ~ omega_{1-alpha}(t) on [dt_min, T]
    pass

def caputo_alikhanov_fast(v_at_grid, t_grid, alpha, nu, s):
    # v_at_grid: [K_t+1, ...], returns (C_0 d_t^alpha v)_{k-theta} for k=1..K_t using SOE recurrence
    pass

net = XfNet()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)
nu, s = soe_weights(alpha=0.5, T=1.0, dt_min=float(t_grid[1] - t_grid[0]))

def total_loss(params, X_col, t_grid, lam, nu, s, alpha):
    v_grid = jnp.stack([hard_ansatz(params, jnp.concatenate(
        [X_col, t_k * jnp.ones_like(X_col[..., :1])], -1), phi0_fn, dist_fn)
                        for t_k in t_grid], 0)
    Dt_alpha_v = caputo_alikhanov_fast(v_grid, t_grid, alpha, nu, s)
    N_v = nonlinear_operator(v_grid, X_col, lam)
    return jnp.mean((Dt_alpha_v + N_v - g_fn(X_col, t_grid))**2)

@jax.jit
def step(params, opt_state, X_col, t_grid, lam, nu, s, alpha):
    g = jax.grad(total_loss)(params, X_col, t_grid, lam, nu, s, alpha)
    updates, opt_state = optimizer.update(g, opt_state, params)
    return optax.apply_updates(params, updates), opt_state
```

Hyperparameters: graded mesh `gamma >= 1` (e.g. 2 or 1/alpha), `theta = alpha/2`, SOE tolerance `epsilon = 1e-8` -> `N_q ~ log` term; adaptive activations with scale `n in {1,2}`; Adam + L-BFGS. For inverse problems append `lambda` to trainable Theta.

## Results
Forward and inverse benchmarks across multiple dimensions (incl. 1-D viscous Burgers `N[v] = lambda1 v v_x - lambda2 v_xx`). Global consistency error scales as `tau^{min(gamma * nu, 2)} + epsilon` (Theorem 1). The accelerated SOE version reduces memory from `O(M K_t)` to `O(M log K_t)` and operations from `O(M K_t^2)` to `O(M K_t log K_t)`, yielding significant CPU-time savings vs uniform-mesh fPINN baselines while preserving second-order accuracy near `t = 0`.
