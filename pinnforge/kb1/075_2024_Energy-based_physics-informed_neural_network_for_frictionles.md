---
slot: 75
title: "Energy-based physics-informed neural network for frictionless contact problems under large deformation"
authors: [Jinshuai Bai, Zhongya Lin, Yizheng Wang, Jiancong Wen, Yinghua Liu, Timon Rabczuk, YuanTong Gu, Xi-Qiao Feng]
year: 2024
venue: "Computer Methods in Applied Mechanics and Engineering (arXiv:2411.03671)"
gitrepo: "https://github.com/JinshuaiBai/energy_PINN_Contact"
---

## TL;DR
The deep energy method (DEM) is extended to **frictionless large-deformation contact** by adding an exponential Lennard-Jones-like surface-contact potential to the total potential energy. Robustness is achieved by relaxation (warm-up without contact term), gradual-loading of displacement BCs, output scaling, and a hard-BC distance-function ansatz.

## Problem
Hertz-type and large-deformation contact problems require enforcing non-penetration inequality constraints. Strong-form PINNs need explicit KKT handling; existing energy-based PINNs do not allow new contact areas to form or surfaces to slide during training.

## Method
Minimise the augmented potential `Pi = E_in - E_ex + E_c` where `E_in = int_Omega Psi(F) dOmega` is the strain energy (Saint-Venant / Neo-Hookean), `E_ex = int_Gamma_t t.u dGamma`, and the surface contact potential is built from a smooth, monotonically decreasing repulsive kernel:
$$ \phi(r) = \phi_0 \exp(-r/r_0),\quad E_c = \iint_{\Gamma_1\times\Gamma_2}\beta_1\beta_2\,\phi(r)\,d\Gamma_1 d\Gamma_2 $$
Two contact discretisations: point-to-point (PP, generic) and point-to-surface (PS, faster for regular surfaces, distance along outward normal).

Essential BCs imposed *hard* via a distance ansatz `u(x) = xi * (F(x;theta) odot g(x) + u_bar(x))` with output scaling `xi`; displacement load uses *gradual* soft-BC `Pi_EBC(t) = (t/t_max) * (1/n) sum ||u - u_bar||^2`.

Training is treated as a pseudo-dynamic relaxation; the learning rate `eta` is selected so the pseudo-velocity `du/dt = -(dF/dtheta)^T dPi/dtheta` produces an increment smaller than the gap.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class DEM(nn.Module):
    H: int = 30
    depth: int = 3
    scale: float = 1.0
    out_dim: int = 2
    @nn.compact
    def __call__(self, x, g, u_bar):                # hard BC
        h = x
        for _ in range(self.depth):
            h = nn.tanh(nn.Dense(self.H)(h))
        f = nn.Dense(self.out_dim)(h)
        return self.scale * (f * g + u_bar)

def strain_energy_point(params, x, mu, lam):       # Neo-Hookean
    u_of = lambda xi: DEM().apply(params, xi, g(xi), u_bar(xi))
    grad_u = jax.jacrev(u_of)(x)                    # (2, 2)
    F = jnp.eye(2) + grad_u
    C = F.T @ F
    J = jnp.linalg.det(F); I1 = jnp.trace(C)
    return 0.5*mu*(I1 - 2) - mu*jnp.log(J) + 0.5*lam*jnp.log(J)**2

def contact_energy(x1, x2, phi0, r0, beta1, beta2):
    diff = x1[:, None, :] - x2[None, :, :]          # (n, m, 2)
    r = jnp.linalg.norm(diff, axis=-1).clip(1e-8)
    phi = phi0 * jnp.exp(-r / r0)
    return jnp.mean(beta1 * beta2 * phi)            # surface-integrated

def total_loss(params, x_int, x_t, traction, surf1, surf2, load, epoch,
               mu, lam, phi0, r0, b1, b2, Omega, Gamma_t, kappa, relax_epochs):
    E_in = jnp.mean(jax.vmap(strain_energy_point, in_axes=(None,0,None,None))(
                    params, x_int, mu, lam)) * Omega
    u_t = jax.vmap(lambda xi: DEM().apply(params, xi, g(xi), u_bar(xi)))(x_t)
    E_ex = jnp.mean(jnp.sum(traction * u_t, axis=-1)) * Gamma_t
    E_c  = jax.lax.cond(epoch >= relax_epochs,
                        lambda _: contact_energy(surf1, surf2, phi0, r0, b1, b2),
                        lambda _: 0.0, None)
    return (E_in - E_ex + E_c) + kappa * load * ebc_loss(params)

optimizer = optax.adam(1e-4); opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, epoch, load):
    grads = jax.grad(total_loss)(params, x_int, x_t, traction, surf1, surf2,
                                  load, epoch, mu, lam, phi0, r0, b1, b2,
                                  Omega, Gamma_t, kappa, relax_epochs)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state

for epoch in range(N):
    load = min(1.0, epoch / loading_epochs)         # gradual loading
    params, opt_state = train_step(params, opt_state, epoch, load)
```

Hyperparameters: tanh MLP 3-5 x 30; Adam lr `1e-4`; `phi0` in `[1e2, 1e6]`, `r0 ~ 1e-5` (problem-dependent); `relax_epochs ~ 2e3`; output scale `xi ~ O(deformation magnitude)`. PP for arbitrary, PS for flat rigid plates.

## Results
On the 2-D Hertz benchmark, predicted contact pressure and stress match the analytical solution closely. The framework solves rubber ironing, ring instability, and double-ring compression - capturing post-buckling and large sliding contact - with runtimes competitive with commercial FEM.
