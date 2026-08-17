---
slot: 119
title: "On Physics-Based Loss Scaling for MF-PINNs applied to the neutron diffusion equation"
authors: [Minh-Hieu Do, Francois Madiot, Karim Ammar, Nicolas Castaing]
year: 2026
venue: arXiv:2604.25957
gitrepo: "https://github.com/ML-SERMA/PINNs"
---

## TL;DR
For Mixed-Formulation PINNs (MF-PINNs) on the multigroup neutron diffusion equation, the authors precondition the standard L2 residual loss by the inverse removal cross-section `δ_e^{-1}` (current residual) and the diffusion coefficient `D` (flux residual). The "Physics-Based Loss Scaling" (PBLS) is norm-equivalent to the unscaled loss but balances per-group gradients, accelerating convergence and improving accuracy with no extra learnable weights.

## Problem
The mixed dual form of the multigroup neutron diffusion equation produces a first-order system with G coupled flux residuals and G·d coupled current residuals. Material cross sections (D, Σ_r, Σ_s) jump by orders of magnitude across reactor regions (fuel/absorber/moderator), so the standard MF-PINN loss `‖div p + T_e φ − S‖² + ‖D⁻¹p + ∇φ‖²` puts hugely uneven gradients on different groups/terms, and adaptive weighting schemes (LRA/NTK/AL/RBA) add training cost or instability.

## Method

### A. Scaled loss (PBLS)
Take the L2-residual of the first-order system and precondition each block with its physical scale. Let `δ_e = diag(Σ_r^g)` (diagonal of the removal matrix). The scaled empirical loss is

$$\mathcal{J}[\zeta]=\|\delta_e^{-1/2}(\mathrm{div}\,p+T_e\phi-S_f)\|^2_{L^2(\Omega)}+\|D^{1/2}(D^{-1}p+\nabla\phi)\|^2_{L^2(\Omega)}.$$

Justified by a posteriori mixed-FEM estimates: the scaled norm is norm-equivalent to the unscaled L2 (Remark 3) so the minimiser is unchanged, yet the per-group gradient magnitudes are dimensionally homogenised.

### B. Hard boundary conditions
For rectangular reactor cores both Dirichlet (`φ=0`) and Neumann/Robin (`p·n=0` or `p·n=φ/2`) become Dirichlet on the appropriate component. Enforce via `φ̃(x)=L(x)φ_θ(x)+B(x)` and analogous for p so no boundary penalty is needed.

### C. k-eigenvalue solve via inverse power iteration
Outer: update `S_f^{n+1}=M_f φ^{n+1}` and `k_{eff}^{n+1}=k_{eff}^n · ⟨f^{n+1},f^{n+1}⟩/⟨f^{n+1},f^n⟩`. Inner: J=2000 Adam steps minimising scaled loss.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class MFNet(nn.Module):
    d: int
    G: int
    hidden: int = 64
    depth: int = 5
    @nn.compact
    def __call__(self, x):
        h = x
        for _ in range(self.depth):
            h = jnp.sin(nn.Dense(self.hidden)(h))            # or jnp.tanh
        out = nn.Dense(self.G * (self.d + 1))(h)
        out = out.reshape(out.shape[:-1] + (self.G, self.d + 1))
        phi, p = out[..., 0], out[..., 1:]
        return phi, p

def hard_bc(x, phi, p, L_fn, B_fn):                          # rectangle: L=x*(1-x)*y*(1-y)...
    return L_fn(x) * phi + B_fn(x), p

def scaled_loss(params, x_r, D, Te_inv_diag, Te, Sf, L_fn, B_fn,
                Mf=None, keff=1.0):
    def phi_fn(x):
        phi, p = net.apply(params, x)
        phi, p = hard_bc(x, phi, p, L_fn, B_fn)
        return phi, p
    phi, p = phi_fn(x_r)
    # ∇φ_g per group g via jacrev over inputs
    grad_phi = jax.vmap(jax.jacrev(lambda y: phi_fn(y)[0]))(x_r)  # (N, G, d)
    # div p_g = Σ_k ∂_k p[g,k]
    def div_pg(y, g):
        jac = jax.jacrev(lambda z: phi_fn(z)[1][g])(y)        # (d, d)
        return jnp.trace(jac)
    div_p = jax.vmap(lambda y: jnp.stack([div_pg(y, g) for g in range(net.G)]))(x_r)
    res_curr = jnp.sqrt(D) * (p / D[..., None] + grad_phi)
    src = Sf if Mf is None else (phi @ Mf.T) / keff
    res_flux = jnp.sqrt(Te_inv_diag) * (div_p + phi @ Te.T - src)
    return jnp.mean(res_curr ** 2) + jnp.mean(res_flux ** 2)

net = MFNet(d=2, G=1)
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
opt = optax.chain(optax.scale_by_adam(),
                  optax.scale_by_schedule(optax.exponential_decay(1e-3, 2000, 0.05)),
                  optax.scale(-1.0))
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, x_r, D, Te_inv, Te, Sf, L_fn, B_fn):
    loss, grads = jax.value_and_grad(scaled_loss)(params, x_r, D, Te_inv, Te, Sf, L_fn, B_fn)
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss
```

Hyperparameters: FCNN L=5, N_l=64, Sin activation + Sobol sampling (best of {tanh-random, sin-sobol}). Source: lr=1e-3, decay 0.05 each 2000. Eigenvalue: lr=2e-4, decay 0.1 each 10000, J=2000 inner iters, stop ε_φ=1e-5, ε_k=1e-6. N_r=10240 (2D) - 200x1024 (3D).

## Results
On IAEA pool reactor (1-group, 2D), simplified C5G7 (2-group), TWIGL-2D/3D k-eigenvalue: scaled loss reduces Δk_eff from ~30-100 pcm to ≤4 pcm and cuts flux relative error 2-5x vs unscaled. Outer iterations drop ~30-40% (e.g. TWIGL-3D 306→185). Sin+Sobol+SL is uniformly the best combination.
