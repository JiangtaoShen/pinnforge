---
slot: 048
title: "SPINN: Sparse, Physics-based, and partially Interpretable Neural Networks for PDEs"
authors: [A. A. Ramabathiran, P. Ramachandran]
year: 2021
venue: Journal of Computational Physics (arXiv:2102.13037)
gitrepo: "https://github.com/nn4pde/SPINN"
---

## TL;DR
SPINN rewrites a radial-basis-function meshless ansatz `u(x) = Σ_i U_i φ((x-X_i)/h_i)` as a sparse DNN whose first ("mesh encoding") layers are hard-wired to compute the scaled distances `|x-X_i|/h_i` and whose nonlinearity is a small "kernel network" shared across nodes. Node positions `X_i`, widths `h_i` and coefficients `U_i` become interpretable trainable parameters, while the kernel net generalises any RBF — so SPINN is a bridge between PINNs and classical meshless solvers.

## Problem
PINN MLPs are over-parameterised and uninterpretable, while classical RBF meshless methods are interpretable but fixed-form. The arbitrary architectural choices in PINNs (width, depth, basis) lack a principled link to the physical discretisation.

## Method
**Sparse architecture (Fig. 1 of paper).** For domain `Ω ⊂ R^d` and `N` "nodes" `X_i ∈ R^d` with widths `h_i > 0`:

1. *Mesh-encoding layer 1* — fixed-weight linear: produces `(x_k - X_{i,k})/h_i` for `k=1..d, i=1..N` (Nd outputs); activation `sqr(z)=z²`.
2. *Mesh-encoding layer 2* — sum the `d` squared coordinates per node and apply `sqrt`, yielding `r_i = ‖x - X_i‖/h_i`.
3. *Kernel layer* — apply a small shared kernel network `φ_θ(r_i)` (any differentiable activation; equivalent to RBF when `φ_θ` is fixed).
4. *Output layer* — linear combination with coefficients `U_i`. Optional partition-of-unity normalisation:
$$ u(x) = \frac{\sum_i U_i \,\varphi_\theta\!\left(\tfrac{\|x-X_i\|}{h_i}\right)}{\sum_j \varphi_\theta\!\left(\tfrac{\|x-X_j\|}{h_j}\right)} $$

Trainable parameters: `{X_i, h_i, U_i}` (interpretable) and the small kernel-net weights `θ`. Loss is the standard PINN composite (PDE residual + BC, computed by autograd through the network). Because `X_i, h_i` are trainable, mesh adaptivity is automatic.

**1-D Fourier variant.** Replace the kernel by `cos(ω_i x + φ_i)` with trainable `ω_i, φ_i, U_i`, recovering Fourier series; replace `cos` with another tiny net for "generalised Fourier" features.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class KernelNet(nn.Module):
    hidden: int = 8
    depth:  int = 2
    @nn.compact
    def __call__(self, r):                              # r: (..., 1)
        for _ in range(self.depth):
            r = jnp.tanh(nn.Dense(self.hidden)(r))
        return nn.Dense(1)(r)                           # φ_θ(r)

class SPINN(nn.Module):
    d: int
    N: int
    partition_of_unity: bool = True

    @nn.compact
    def __call__(self, x):                              # x: (B, d)
        side = int(round(self.N ** (1.0 / self.d)))
        grid = jnp.stack(jnp.meshgrid(*[jnp.linspace(0, 1, side)] * self.d,
                                       indexing='ij'), axis=-1).reshape(-1, self.d)
        X = self.param('X', lambda _: grid)             # (N, d)
        h = self.param('h', lambda _: jnp.full((self.N,), 1.0 / side))  # (N,)
        U = self.param('U', lambda _: jnp.zeros(self.N))                # (N,)
        diff = (x[:, None, :] - X[None, :, :]) / h[None, :, None]
        r    = jnp.linalg.norm(diff, axis=-1, keepdims=True)            # (B, N, 1)
        phi  = KernelNet()(r).squeeze(-1)                               # (B, N)
        num  = (phi * U).sum(axis=1)                                    # (B,)
        if self.partition_of_unity:
            den = phi.sum(axis=1) + 1e-12
            return (num / den)[:, None]
        return num[:, None]

model = SPINN(d=2, N=400)                              # 400 nodes
params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
opt    = optax.adam(1e-3)
opt_state = opt.init(params)

def loss_fn(params, x_f, x_b, u_b):
    def u_of(xx): return model.apply(params, xx)
    u_x = jax.grad(lambda xx: u_of(xx).sum())(x_f)
    res = pde_op(u_of(x_f), u_x, x_f)
    return jnp.mean(res ** 2) + jnp.mean((u_of(x_b) - u_b) ** 2)

@jax.jit
def step(params, opt_state, x_f, x_b, u_b):
    g = jax.grad(loss_fn)(params, x_f, x_b, u_b)
    upd, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state

for it in range(20_000):
    params, opt_state = step(params, opt_state, x_f, x_b, u_b)
```

Recommended setup: `N` chosen like a meshless discretisation (e.g. 200–1000 nodes for 1-D/2-D); kernel net is a small (1→8→8→1) tanh MLP; Adam lr=1e-3 → L-BFGS for refinement; init `X_i` on a uniform grid, `h_i` at the spacing, `U_i = 0`. Node positions adapt to discontinuities during training; you may add an L1 penalty on `U_i` for sparsity.

## Results
Tested on a wide range of ODEs and PDEs: elliptic (Poisson), parabolic (heat), hyperbolic (transport with discontinuities) and a fluid-dynamics example. SPINN gives accuracy comparable to or better than PINN with far fewer trainable parameters (because the input layer is hard-wired), provides interpretable node positions and widths that adapt to solution features, and yields a natural Fourier-net analogue in 1-D.
