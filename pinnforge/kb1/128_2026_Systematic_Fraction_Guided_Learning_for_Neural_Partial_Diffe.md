---
slot: 128
title: "Systematic Fraction Guided Learning for Neural Partial Differential Equation Solvers"
authors: [Mahboubeh Molavi-Arabshahi, Khalid Sumiea Munshid]
year: 2026
venue: 2026 DCHPC
gitrepo: ""
doi: 10.1109/DCHPC69296.2026.11517242
---

## TL;DR
Fractional Physics-Informed Neural Network (FPINN) framework for fractional PDEs with Caputo derivatives. Three integrated pieces: (i) Mittag-Leffler activation `E_α(x)` to suppress spectral bias toward low frequencies of fractional operators, (ii) FFT-based spectral evaluation of the fractional derivative to drop complexity from O(N²) to O(N log N), (iii) learnable per-layer fractional order `α_ℓ` jointly optimized with weights.

## Problem
Fractional differential equations (long memory, nonlocal dynamics) are nontrivial for vanilla PINNs: standard tanh/ReLU activations don't match the long-tail behavior of `E_α`; AD on the Caputo integral has O(N²) cost; fixed fractional orders are suboptimal across regions of the domain.

## Method

### A. Caputo time-fractional operator
$$D_t^\alpha u(x,t) = \frac{1}{\Gamma(1-\alpha)}\int_0^t \frac{\partial_\tau u(x,\tau)}{(t-\tau)^\alpha}\,d\tau,\quad 0<\alpha<1$$
Evaluated via spectral product `F[L_α u](ξ) = (iω)^α F[u](ξ)` + inverse FFT → `O(N log N)`.

### B. Mittag-Leffler activation
$$E_\alpha(x)=\sum_{k=0}^\infty \frac{x^k}{\Gamma(\alpha k + 1)}$$
Replaces tanh in hidden layers (truncated to ~30 terms). Empirically converges to 10⁻⁴ in 1200 iterations vs 1800-2100 for tanh/ReLU.

### C. Learnable per-layer fractional order
Each hidden layer `ℓ` carries trainable `α_ℓ ∈ (0,1]`. The fractional layer transform:
$$F_{\alpha_\ell}[z_\ell](t) = \frac{1}{\Gamma(1-\alpha_\ell)}\int_0^t \frac{\partial_\tau z_\ell(\tau)}{(t-\tau)^{\alpha_\ell}}\,d\tau$$
implemented as a discrete convolution whose kernel weights are differentiable w.r.t. `α_ℓ` via the spectral form.

### D. Loss & training
Standard PINN composite:
$$J(\theta)=\frac1{N_f}\sum_i |L_\alpha[u_\theta(x_i,t_i)]-f(x_i,t_i)|^2 + \frac{\lambda}{N_b}\sum_j|u_\theta(x_j,t_j)-g(x_j,t_j)|^2$$
Use fractional-Adam (Caputo-derivative momentum) for stability across nonlocal updates.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from jax.scipy.special import gammaln

def gamma(x): return jnp.exp(gammaln(x))

class MittagLeffler(nn.Module):
    alpha_init: float = 0.8
    K: int = 20
    learnable: bool = True
    @nn.compact
    def __call__(self, x):
        if self.learnable:
            alpha = self.param("alpha", lambda k: jnp.array(self.alpha_init))
        else:
            alpha = self.alpha_init
        out, xpow = jnp.zeros_like(x), jnp.ones_like(x)
        for k in range(self.K):
            out = out + xpow / gamma(alpha * k + 1.0)
            xpow = xpow * x
        return out

def caputo_spectral(u, t, alpha):
    """Caputo time-fractional derivative via FFT spectral product."""
    N = u.shape[-1]
    dt = (t[..., -1] - t[..., 0]) / (N - 1)
    omega = jnp.fft.fftfreq(N, d=float(dt)) * 2 * jnp.pi
    U = jnp.fft.fft(u, axis=-1)
    factor = (1j * omega + 1e-12) ** alpha
    Du = jnp.real(jnp.fft.ifft(factor * U, axis=-1))
    return Du

class FPINN(nn.Module):
    hidden: int = 128
    depth: int = 4
    @nn.compact
    def __call__(self, x):
        h = x
        for _ in range(self.depth):
            h = nn.Dense(self.hidden)(h)
            h = MittagLeffler(alpha_init=0.8)(h)
        return nn.Dense(1)(h)

def fpinn_loss(params, X_int, X_bc, f_fn, g_fn, alpha=0.8, lam=10.0):
    def u_fn(x): return net.apply(params, x).squeeze()
    u = jax.vmap(u_fn)(X_int)
    t = X_int[..., -1:]
    Du = caputo_spectral(u[..., None], t[..., None], alpha).squeeze(-1)
    res = Du - laplacian_x(params, X_int) - f_fn(X_int)
    Lr = jnp.mean(res ** 2)
    Lb = jnp.mean((jax.vmap(u_fn)(X_bc) - g_fn(X_bc)) ** 2)
    return Lr + lam * Lb

net = FPINN()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
opt = optax.adam(optax.linear_schedule(1e-3, 0.0, 50_000))
opt_state = opt.init(params)
```

Hyperparameters: 4-hidden-layer MLP x 128 units, Mittag-Leffler activations, Caputo `α∈[0.8,0.95]`, Adam lr=1e-3 → 0 over 50k iterations, 1000 collocation points, 5 random seeds. NVIDIA A100.

## Results
On fractional diffusion-transport: FPINN reaches relative accuracy 99.3% (avg over 5 seeds, std 0.4%) in 12 s, vs 96.1% (standard PINN, 30 s), 97.5% (spectral solver, 28 s), 94.2% (Dense NN, 32 s). On fracture-mechanics crack propagation: 98.7% accuracy, ~70% wall-time savings vs FEM. Hydrology groundwater modeling: 97.3% accuracy, 45% better pollutant transport. Computational complexity `O(N log N)` vs `O(N²)` for FDM/FEM.
