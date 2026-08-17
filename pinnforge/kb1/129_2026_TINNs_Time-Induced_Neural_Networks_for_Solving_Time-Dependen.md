---
slot: 129
title: "TINNs: Time-Induced Neural Networks for Solving Time-Dependent PDEs"
authors: [Chen-Yang Dai, Che-Chia Chang, Te-Sheng Lin, Ming-Chih Lai, Chieh-Hsin Lai]
year: 2026
venue: arXiv:2601.20361
gitrepo: ""
---

## TL;DR
Standard space-time PINNs `u_θ(x,t)` reuse one parameter set across all times — time only enters as an input shift, so steepening spatial gradients (e.g. Burgers shock) is hard to represent. TINNs replace this with `u_{θ(t)}(x)`: a small "time hyper-network" outputs a layer-wise time code `Φ(t) ∈ R^{2L}` that is affinely lifted to the full backbone parameter vector `θ(t)`, giving an explicit time-evolving spatial network. Trained with Levenberg-Marquardt nonlinear least squares.

## Problem
For a vanilla MLP `u_θ(x,t) = U(wx + vt + b)`, the spatial derivative is `w · U'(wx + vt + b)` — time can only translate `U'`'s argument, never rescale `w`. The same "time-entanglement" affects ResNets, Fourier-feature networks, and SPINN. This makes shock-developing PDEs (Burgers) and stiff problems (Allen-Cahn) hard.

## Method

### A. Time-induced parameterisation
$$u(x,t) = u_{\theta(t)}(x)$$
Backbone MLP `u_θ : Ω → R` (2 hidden layers x 20 units, tanh). Time enters by modulating `θ(t)`.

### B. Compact layer-wise time embedding
A small time net `N : R → R^{2L}` (`L` = number of parameter groups, e.g. 5 for a 2-layer MLP: W1, b1, W2, b2, W3). Output the layer-wise code via a learnable gate:
$$\Phi(t) = (1 - \boldsymbol\alpha)\,t + \boldsymbol\alpha \odot \mathcal N(t)$$
Lift to full parameter vector `θ(t) ∈ R^{N_D}` via an entrywise affine map: for each backbone weight `w^{ℓ}_{ij}`,
$$w^{\ell}_{ij}(t) = a^{\ell}_{ij}\,\Phi_{2\ell-1}(t) + b^{\ell}_{ij}$$
Each parameter group shares one coordinate of `Φ(t)` (macro coherence) but each entry has its own `(a,b)` (micro diversity). Total params `2 N_D + O(L h)` rather than `O(N_D h)` of a naïve hypernet.

### C. Optimisation: Levenberg-Marquardt
The PINN loss is naturally NLS:
$$L = \lambda_r \|L u_{\theta(t)}\|^2 + \lambda_b \|B u_{\theta(t)}\|^2 + \lambda_{ic} \|I u_{\theta(t)}\|^2$$
Use LM update on the stacked residual `r(ψ)`:
$$\psi \leftarrow \psi - (J^\top J + \mu I)^{-1} J^\top r$$
with adaptive damping `μ`. Tractable because TINN keeps the parameter count small.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class TimeNet(nn.Module):
    n_groups: int = 5
    hidden: int = 10
    depth: int = 2
    @nn.compact
    def __call__(self, t):                                   # t: (B, 1)
        h = jnp.tanh(nn.Dense(self.hidden)(t))
        for _ in range(self.depth - 1):
            h = jnp.tanh(nn.Dense(self.hidden)(h))
        nn_out = nn.Dense(2 * self.n_groups)(h)
        alpha = self.param("alpha", lambda k: jnp.zeros(2 * self.n_groups))
        return (1 - alpha) * t + alpha * nn_out

SHAPES = ((20, 1), (20,), (20, 20), (20,), (1, 20))           # (W1, b1, W2, b2, W3)

class TINN(nn.Module):
    @nn.compact
    def __call__(self, x, t):                                # x: (B, d_in), t: (B, 1)
        a_list = [self.param(f"a{ℓ}", nn.initializers.normal(0.01), s) for ℓ, s in enumerate(SHAPES)]
        b_list = [self.param(f"b{ℓ}", nn.initializers.normal(0.01), s) for ℓ, s in enumerate(SHAPES)]
        phi = TimeNet(n_groups=len(SHAPES))(t)               # (B, 2*L)
        params_at = []
        for ℓ, (a, b) in enumerate(zip(a_list, b_list)):
            c = phi[..., 2*ℓ:2*ℓ+1]                          # (B, 1)
            shape = (-1,) + (1,) * a.ndim
            params_at.append(a * c.reshape(shape) + b)        # (B, *a.shape)
        W1, b1, W2, b2, W3 = params_at
        h = jnp.tanh(jnp.einsum('boi,bi->bo', W1, x) + b1)
        h = jnp.tanh(jnp.einsum('boi,bi->bo', W2, h) + b2)
        return jnp.einsum('boi,bi->bo', W3, h).squeeze(-1)

def residual_vector(params, X, T):
    u_pred = net.apply(params, X, T)
    return jnp.concatenate([(L_op(u_pred, X, T)).ravel(),
                            (B_op(u_pred, X, T)).ravel(),
                            (I_op(u_pred, X, T)).ravel()])

def lm_step(params, X, T, mu=1e-3):
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    def r_of_flat(v): return residual_vector(unravel(v), X, T)
    r = r_of_flat(flat)
    J = jax.jacrev(r_of_flat)(flat)                          # (M, P)
    JTJ = J.T @ J
    delta = jnp.linalg.solve(JTJ + mu * jnp.eye(JTJ.shape[0]), J.T @ r)
    return unravel(flat - delta)

net = TINN()
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 1)), jnp.zeros((1, 1)))
```

Hyperparameters: spatial backbone 2 hidden x 20 tanh; time-net 2 hidden x 10; ~1145 total params (vs ~310k for vanilla PINN, 535k for PirateNet). LM optimizer with adaptive damping for ≤30k iterations on RTX A6000.

## Results
Five benchmarks (Burgers, Allen-Cahn, Klein-Gordon, KdV, Wave). TINN relative L2: Burgers 6.89e-7 (PirateNet+SOAP 1.97e-6, 2.9x better), Allen-Cahn 3.85e-6 (8.32e-6, 2.2x), KdV 1.53e-4 (4.26e-4, 2.8x), Wave 6.71e-6 (2.88e-5, 4.3x). Training-time speedup over PirateNet+SOAP: 10.55x on Burgers, 2.30x on Allen-Cahn. Uses ~470x fewer parameters than PirateNet.
