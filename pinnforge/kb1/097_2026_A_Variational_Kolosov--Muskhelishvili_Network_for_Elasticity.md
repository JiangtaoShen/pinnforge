---
slot: 97
title: "A Variational Kolosov-Muskhelishvili Network for Elasticity and Fracture"
authors: [Shuwei Zhou, Christian Haffner, Sophie Stebner, Niklas Fehlemann, Zhichao Wei, Sebastian Munstermann]
year: 2026
venue: arXiv:2605.02310
gitrepo: ""
---

## TL;DR
For 2-D isotropic linear elastic (and fracture) problems, replace the displacement-output PINN by two complex-valued holomorphic neural nets that output the Kolosov-Muskhelishvili (KM) potentials `phi(z), psi(z)`. Train them by minimizing the total potential energy (deep energy method) with one Dirichlet penalty -- no PDE residual, no traction loss. For cracks, use the discontinuous stress potential `phi = zeta(z) F1 + F2`, `omega = zeta(z) F1* - F2*` with `zeta = sqrt(z^2 - a^2)`, so the `sqrt(r)` singularity and traction-free crack faces are built in.

## Problem
Standard PINNs for elasticity output `u(x,y)` and form `sigma` by autograd + Hooke, then enforce equilibrium + traction + Dirichlet as a multi-term residual loss. For cracks they require Williams enrichment and a separate traction-free-face loss; balancing all of these is unstable. KM-INN residual variants help but still have many loss terms; no variational KM formulation existed.

## Method
Stresses and displacements come from two holomorphic potentials phi(z), psi(z):
$$
\sigma_{xx}+\sigma_{yy}=4\mathrm{Re}\,\phi'(z),\quad
\sigma_{yy}-\sigma_{xx}+2i\sigma_{xy}=\bar z\phi''(z)+\psi'(z)
$$
$$
2\mu(u_x+iu_y)=\kappa\,\phi(z)-z\overline{\phi'(z)}-\overline{\psi(z)},\quad
\kappa=\tfrac{3-\nu}{1+\nu}\,\,(\text{plane stress})
$$

Energy-based loss (deep energy method):
$$
\Psi[\phi,\psi]=\int_\Omega w(\sigma)\,d\Omega - \int_{\Gamma_t}\bar t\cdot u\,d\Gamma + \alpha_u\int_{\Gamma_u}\|u-\bar u\|^2 d\Gamma
$$
$$
w(\sigma)=\frac1{4\mu}\big(\sigma_{xx}^2+\sigma_{yy}^2+2\sigma_{xy}^2\big)-\frac{\nu}{2E}(\sigma_{xx}+\sigma_{yy})^2
$$
Integrals are Monte-Carlo with `|Omega|` known analytically or estimated by bounding-box rejection.

Crack ansatz (Woo): with `omega = psi + z phi'`,
$$
\phi_{\text{pred}}=\zeta(z)\,N_{F_1}(z)+N_{F_2}(z),\quad
\omega_{\text{pred}}=\zeta(z)\,N_{F_1}^\star(z)-N_{F_2}^\star(z)
$$
$$
\zeta(z)=\sqrt{z^2-a^2}\;(\text{internal})\text{ or }\sqrt z\;(\text{edge})
$$
gives traction-free crack faces for free and reproduces `1/sqrt(r)` stress singularity. SIFs `K_I, K_II` are recovered by the interaction integral.

Architecture: two parallel complex-valued MLPs with entire activation `sigma(xi) = exp(xi)` (preserves holomorphicity) and exponential-aware He-style init.

```python
import jax, jax.numpy as jnp
import flax.linen as nn

class CLinear(nn.Module):
    n_in: int; n_out: int
    @nn.compact
    def __call__(self, z):                                 # z: complex (B, n_in)
        Wr = self.param("Wr", nn.initializers.normal(1e-2), (self.n_out, self.n_in))
        Wi = self.param("Wi", nn.initializers.normal(1e-2), (self.n_out, self.n_in))
        br = self.param("br", nn.initializers.zeros, (self.n_out,))
        bi = self.param("bi", nn.initializers.zeros, (self.n_out,))
        W = Wr + 1j*Wi; b = br + 1j*bi
        return z @ W.T + b

class HoloNet(nn.Module):
    depth: int = 4; hidden: int = 32
    @nn.compact
    def __call__(self, z):
        x = z[..., None]
        x = CLinear(1, self.hidden)(x)
        for _ in range(self.depth - 1):
            x = jnp.exp(CLinear(self.hidden, self.hidden)(x))
        x = CLinear(self.hidden, 1)(x)
        return x.squeeze(-1)

phi_net = HoloNet(); psi_net = HoloNet()                  # crack case: F1_net, F2_net
params_phi = phi_net.init(jax.random.PRNGKey(0), jnp.zeros((1,), dtype=jnp.complex64))
params_psi = psi_net.init(jax.random.PRNGKey(1), jnp.zeros((1,), dtype=jnp.complex64))

def complex_grad(net, params, z):
    # holomorphic derivative phi'(z): differentiate Re and Im separately
    fr = lambda zv: net.apply(params, zv).real
    fi = lambda zv: net.apply(params, zv).imag
    return jax.vmap(jax.grad(fr))(z) + 1j * jax.vmap(jax.grad(fi))(z)

def fields(z, params_phi, params_psi, kappa, mu, E, nu):
    phi  = phi_net.apply(params_phi, z)
    psi  = psi_net.apply(params_psi, z)
    dphi = complex_grad(phi_net, params_phi, z)
    ddphi= complex_grad(lambda p, zz: complex_grad(phi_net, p, zz),
                         params_phi, z)                   # second derivative
    dpsi = complex_grad(psi_net, params_psi, z)
    sxx = (2*dphi - jnp.conj(z)*ddphi - dpsi).real
    syy = (2*dphi + jnp.conj(z)*ddphi + dpsi).real
    sxy = (jnp.conj(z)*ddphi + dpsi).imag
    U   = (kappa*phi - z*jnp.conj(dphi) - jnp.conj(psi))/(2*mu)
    return sxx, syy, sxy, U.real, U.imag

def energy_loss(params_phi, params_psi, z_int, z_t, z_u, t_bar, u_bar,
                kappa, mu, E, nu, AREA, LEN_T, LEN_U, alpha_u=1000.0):
    sxx, syy, sxy, _, _ = fields(z_int, params_phi, params_psi, kappa, mu, E, nu)
    w = (sxx**2 + syy**2 + 2*sxy**2)/(4*mu) - nu/(2*E)*(sxx+syy)**2
    W_int = AREA * jnp.mean(w)
    _, _, _, ux_t, uy_t = fields(z_t, params_phi, params_psi, kappa, mu, E, nu)
    W_ext = LEN_T * jnp.mean(t_bar[:, 0]*ux_t + t_bar[:, 1]*uy_t)
    _, _, _, ux_u, uy_u = fields(z_u, params_phi, params_psi, kappa, mu, E, nu)
    L_u  = LEN_U * jnp.mean((ux_u - u_bar[:, 0])**2 + (uy_u - u_bar[:, 1])**2)
    return W_int - W_ext + alpha_u * L_u
```

Hyperparameters: 4-layer complex MLPs of width 32 with `exp` activation; alpha_u = 1000; Adam; exponential-aware He init + early gradient clipping. Sampling: quasi-uniform interior points and arc-length-uniform boundary points; exclude a small disk around each crack tip; refine near the tip for accuracy.

## Results
Three crack-free benchmarks (pressurized circular tube, plate with a hole under tension, plate under sinusoidal traction) plus mode-I and mixed-mode crack panels. vKMINN matches FEM stress / displacement with R^2 > 0.99 and lower L2 error than displacement-output PINN and residual-based PIHNN/KMINN. SIFs from the interaction integral are within ~1-3% of reference values; the discontinuous-stress-potential ansatz removes the need for Williams enrichment and traction-free face losses.
