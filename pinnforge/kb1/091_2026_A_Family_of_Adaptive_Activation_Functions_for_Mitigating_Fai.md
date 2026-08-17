---
slot: 91
title: "A Family of Adaptive Activation Functions for Mitigating Failure Modes in Physics-Informed Neural Networks"
authors: [Krishna Murari]
year: 2026
venue: arXiv:2603.18328
gitrepo: ""
---

## TL;DR
Replace tanh in PINN hidden layers with adaptive wavelet-modulated activations of the form `tanh(beta x) * psi_wavelet(x; alpha, ...)` where all shape parameters are softplus-positively trainable. Five variants (SoftMexTanh, SoftMorTanh, SoftGaussTanh, SoftGaborTanh, SoftHerTanh) fix the well-known PINN failure modes on convection (beta'=50), wave (beta'=3), reaction (rho=5) and 2-D Navier-Stokes.

## Problem
Standard tanh PINNs fail on high-frequency / oscillatory / multi-scale PDEs (Krishnapriyan et al. 2021) due to spectral bias and ill-conditioned gradients. Existing wavelet activations are fixed-shape; the author argues per-neuron adaptive wavelet shape is needed.

## Method
Activations multiply a trainable tanh by a wavelet-inspired window with positive (softplus-parameterized) trainable widths/frequencies:

Mexican-hat tanh:
$$
\psi_{\text{SoftMex}}(x) = \tanh(\beta x)\,(1-\gamma x^2)\,e^{-\alpha x^2}
$$

Morlet tanh, Gaussian tanh, Gabor tanh, Hermite-`n` tanh:
$$
\psi_{\text{SoftMor}} = \tanh(\beta x)\cos(\omega x)e^{-x^2/(2\sigma^2)},\quad
\psi_{\text{SoftGauss}} = \tanh(\beta x)\,e^{-\alpha x^2}
$$
$$
\psi_{\text{SoftGabor}} = \tanh(\beta x)\,e^{-x^2/(2\sigma^2)}\cos(\omega x),\quad
\psi_{\text{SoftHer}_n} = \tanh(\beta x)\,H_n(x)\,e^{-\alpha x^2}
$$
with `alpha = softplus(alpha_0)`, etc. The "-W" variant freezes `beta`. Recommended init: trainable params = 1, except `omega_0 in {3,5}` for Gabor.

Composite loss is the standard `L = L_R + L_B + L_I` (equal weights, all unity), `L_R` is the MSE of the autograd PDE residual at collocation points.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

softplus = jax.nn.softplus

class SoftMorTanh(nn.Module):
    @nn.compact
    def __call__(self, x):
        w0 = self.param("w0", nn.initializers.ones, ())
        s0 = self.param("s0", nn.initializers.ones, ())
        b0 = self.param("b0", nn.initializers.ones, ())
        w, s, b = softplus(w0), softplus(s0), softplus(b0)
        return jnp.tanh(b*x) * jnp.cos(w*x) * jnp.exp(-x**2/(2*s**2))

class SoftGaussTanh(nn.Module):
    @nn.compact
    def __call__(self, x):
        a0 = self.param("a0", nn.initializers.ones, ())
        b0 = self.param("b0", nn.initializers.ones, ())
        a, b = softplus(a0), softplus(b0)
        return jnp.tanh(b*x) * jnp.exp(-a*x*x)

class PINN(nn.Module):
    hidden: int = 512
    depth: int = 4
    out_dim: int = 1
    act_cls: type = SoftGaussTanh
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = self.act_cls()(nn.Dense(self.hidden)(x))
        return nn.Dense(self.out_dim)(x)

net = PINN()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))

def total_loss(params, X_R, X_B, X_I):
    return pde_residual_mse(params, X_R) + bc_mse(params, X_B) + ic_mse(params, X_I)

# L-BFGS via jaxopt (drop-in for strong-Wolfe L-BFGS)
import jaxopt
solver = jaxopt.LBFGS(fun=total_loss, maxiter=1000, linesearch="zoom")
state = solver.init_state(params, X_R, X_B, X_I)

@jax.jit
def lbfgs_step(params, state):
    return solver.update(params, state, X_R, X_B, X_I)

for _ in range(1000):
    params, state = lbfgs_step(params, state)
```

Hyperparameters: 4 hidden layers x 512 neurons, L-BFGS with strong Wolfe line search (1000 iters), all loss weights 1. 1-D: `N_I = N_B = 101`, `N_R = 101^2 = 10201`. 2-D NS: 2500 sampled collocation points. Use the -W (fixed beta) variant for convection beta'=50.

## Results
On 1-D reaction (rho=5): SoftGaborTanh rRMSE 2.1e-3 vs tanh 0.973; on 1-D wave (beta'=3): SoftMorTanh / SoftGaborTanh ~3e-2 vs tanh 0.22; on convection beta'=50 the -W variants beat tanh, PINNsFormer, QRes, FLS, PINN-Mamba, ML-PINN; on 2-D Navier-Stokes the proposed activations produce the lowest pressure error at t=20.0 versus all listed baselines.
