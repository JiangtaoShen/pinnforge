---
slot: 117
title: "MetaColloc: Optimization-Free PDE Solving via Meta-Learned Basis Functions"
authors: [Zichuan Yang]
year: 2026
venue: arXiv:2605.12368
gitrepo: ""
---

<!-- input is a pymupdf fallback (plain text + page markers); content reconstructed from raw flow -->

## TL;DR
Decouple basis discovery from PDE solving. Offline, meta-train a *dual-branch* (low-frequency MLP + multi-scale Fourier-feature) network on diverse Gaussian Random Fields by inner least-squares so the learned features form a universal neural dictionary `Φ(x)`. Online, freeze the network, build a collocation matrix using forward-mode AD derivatives, and solve a single linear least-squares for coefficients `w` (Newton-Raphson for nonlinear PDEs). No test-time gradient descent, no PDE solution data anywhere.

## Problem
PINNs train a fresh network for *every* PDE — slow, spectral-biased, unstable on stiff/nonlinear systems. Operator learners (FNO, DeepONet) need large simulation datasets. Random-feature ELMs solve in closed form but their basis quality is sensitive to random init and ad-hoc shape parameters.

## Method

### A. Dual-branch basis dictionary `Φ_θ : R^d → R^H`
Split `Φ = [φ_low, φ_high]` each of width `H/2`.
- *Low-freq branch*: 2 SwiGLU MLP layers on raw `x`. SwiGLU: `h_k = SiLU(W_1 h_{k-1}+b_1)\odot(W_2 h_{k-1}+b_2)`.
- *High-freq branch*: multi-scale Fourier features
$$
\gamma(x)=\big[\sin(\pi k_j^\top x),\cos(\pi k_j^\top x)\big]_{j=1}^{F},\quad k_j\in\{1,2,4,8,16,32,64,128\}\text{ axis-aligned},
$$
followed by 2 SwiGLU layers.

### B. Meta-training on Gaussian Random Fields (offline, data-free)
For epoch e, task t: sample `X` and `Y` from a multi-scale GRF (length-scales over 2 orders of magnitude, mix smooth radial + oscillatory periodic kernels). Compute `Φ_θ(X)`, solve inner least squares
$$
w=\operatorname{lstsq}\big(\Phi_\theta(X),\,Y\big),\quad \hat Y=\Phi_\theta(X)\,w,
$$
update `θ` by AdamW on `MSE(\hat Y, Y)`. The outer objective drives `Φ` to be a *universally* expressive dictionary.

### C. Online linear PDE solve (frozen `Φ`)
For Poisson `-Δu = f`, `u|_∂ = g`, sample interior `X_int` and boundary `X_bd`. Compute basis and its derivatives via *forward-mode* AD:
$$
A_{\text{eq}}=-(\Phi_{xx}+\Phi_{yy}),\quad A_{\text{bd}}=\Phi_{bd},\quad
\begin{bmatrix}A_{\text{eq}}\\A_{\text{bd}}\end{bmatrix}w=\begin{bmatrix}f\\g\end{bmatrix}.
$$
Solve in one least-squares step. `u(x)=Φ_frozen(x)·w`.

### D. Newton-Raphson for nonlinear PDEs
At iterate `w^{(k)}`, linearise `N[Φw] = 0`: form Jacobian `J^{(k)} = (∂N/∂u)(Φw^{(k)})·Φ_op` where `Φ_op` already encodes spatial derivatives. Solve
$$
J^{(k)}\Delta w=-N[\Phi w^{(k)}],\quad w^{(k+1)}=w^{(k)}+\Delta w,
$$
in 5-8 iters to high accuracy on Sine-Gordon, KdV.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class DualBranchBasis(nn.Module):
    d: int = 2
    H: int = 256
    scales: tuple = (1, 2, 4, 8, 16, 32, 64, 128)

    @staticmethod
    def _swiglu(dense_a, dense_b, x):
        return jax.nn.silu(dense_a(x)) * dense_b(x)

    @nn.compact
    def __call__(self, x):
        # low-freq branch: 2 SwiGLU layers
        l = self._swiglu(nn.Dense(self.H, name="low_a0"),
                         nn.Dense(self.H, name="low_b0"), x)
        l = self._swiglu(nn.Dense(self.H//2, name="low_a1"),
                         nn.Dense(self.H//2, name="low_b1"), l)
        # high-freq Fourier features
        K = jnp.tile(jnp.asarray(self.scales, dtype=jnp.float32)[:, None],
                     (1, self.d))                       # (F, d)
        proj = x @ K.T                                  # (N, F)
        gamma = jnp.concatenate([jnp.sin(jnp.pi*proj), jnp.cos(jnp.pi*proj)], -1)
        h = self._swiglu(nn.Dense(self.H, name="hi_a0"),
                         nn.Dense(self.H, name="hi_b0"), gamma)
        h = self._swiglu(nn.Dense(self.H//2, name="hi_a1"),
                         nn.Dense(self.H//2, name="hi_b1"), h)
        return jnp.concatenate([l, h], axis=-1)         # (N, H)

# Offline: meta-training inner-loop = lstsq, outer = MSE on GRF tasks
@jax.jit
def meta_step(params, opt_state, X, Y, opt):
    def loss_fn(p):
        Phi  = DualBranchBasis().apply(p, X)
        w    = jnp.linalg.lstsq(Phi, Y, rcond=None)[0]
        Yhat = Phi @ w
        return jnp.mean((Yhat - Y)**2)
    g = jax.grad(loss_fn)(params)
    updates, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

# Online: solve -Δu = f with frozen Φ
def solve_poisson(params, X_int, X_bd, f, g):
    apply = lambda x: DualBranchBasis().apply(params, x)
    PhiL_int = laplacian_basis(apply, X_int)            # (N, H), via jax.jacfwd
    Phi_bd   = apply(X_bd)
    A = jnp.concatenate([-PhiL_int, Phi_bd], axis=0)
    b = jnp.concatenate([f.flatten(), g.flatten()])
    w = jnp.linalg.lstsq(A, b, rcond=None)[0]
    return w
```

Hyper-parameters: `H≈256-512`, SwiGLU widths `(2H, H)`, 8 Fourier scales `{1,...,128}`, `optax.adamw(3e-4)`, 100k meta-tasks; GRF kernels Matern + periodic, length-scales `ℓ∈[1e-2, 1.0]`.

## Results
On six 2-D / 3-D benchmark PDEs (Poisson, high-freq Poisson, heat, advection, Sine-Gordon, KdV), MetaColloc matches state-of-the-art PINN accuracy on smooth problems and beats it on nonlinear + high-frequency cases, while reducing test-time wall-clock by **several orders of magnitude** (a single linear solve, vs thousands of PINN gradient steps). Frequency-sweep analysis exposes a remaining gap between function-approximation power of the dictionary and operator stability at extreme high frequencies — motivating future *operator-aware* meta-learning.
