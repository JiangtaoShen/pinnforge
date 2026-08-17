---
slot: 034
title: "DAS-PINNs: A deep adaptive sampling method for solving high-dimensional partial differential equations"
authors: [Kejun Tang, Xiaoliang Wan, Chao Yang]
year: 2021
venue: "J. Comp. Phys. (arXiv:2112.14038)"
gitrepo: ""
---

## TL;DR
DAS-PINN: train a deep generative model (KRnet, a normalizing flow with triangular Knothe-Rosenblatt structure) to fit the residual^2 of the current PINN as a probability density. Sample new collocation points from that flow — points concentrate where the PDE residual is largest — and refine the training set. Treats KRnet as a residual-based posteriori error estimator.

## Problem
Uniform-LHS collocation has a generalization-error prefactor that explodes for solutions of low regularity, and in high-d most uniform volume lives near the boundary. Residual-based MCMC resampling is ad-hoc and doesn't scale. Need a generic high-d residual-adaptive sampler with explicit density.

## Method
Solve Lu = s on Omega = [-1/2, 1/2]^d. Standard PINN loss J_N = ||r(x;Theta)||^2_{N_r,S} + gamma_hat ||b||^2. Two iterated stages:

**Stage A (PINN step).** With current collocation set S, minimize empirical loss via Adam.

**Stage B (KRnet step).** Treat target density
$$
\hat r_X(x) \propto r^2(x;\Theta)\,h(x)
$$
where h is a piecewise-linear cutoff softening Omega's boundary. Fit a KRnet density model
$$
p_{KR}(x;\Theta_f) = p_Z(f_{KR}(x))\,|\det\nabla_x f_{KR}|
$$
with Z standard Gaussian and f_KR an invertible lower-triangular (K-R) flow. Compose with a logarithmic map ell: Omega -> R^d so the support matches Omega. Train Theta_f by minimizing the importance-sampled cross entropy
$$
H \approx -\frac{1}{N_r}\sum_i \frac{\hat r_X(x_i)}{\hat p_{KR}(x_i; \hat\Theta_f)}\log \hat p_{KR}(x_i;\Theta_f)
$$
Then resample S from p_X ~ KRnet restricted to Omega and iterate.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax, math

class KRnet(nn.Module):
    """Lower-triangular normalizing flow z = T^{-1}(x).  Stack of affine couplings."""
    d: int
    n_blocks: int = 8
    hidden: int = 64
    @nn.compact
    def __call__(self, x):
        # returns (z, log|det J|) — implement triangular affine coupling blocks here
        log_det = jnp.zeros(())
        z = x
        for i in range(self.n_blocks):
            z, ld = TriAffineCoupling(d=self.d, hidden=self.hidden,
                                      name=f"b{i}")(z)
            log_det = log_det + ld
        return z, log_det
    def log_prob(self, x):
        z, log_det = self(x)
        return -0.5*(z**2).sum(-1) - 0.5*self.d*math.log(2*math.pi) + log_det
    def inverse(self, params, z):
        # Apply each block in reverse using its `.inverse` method.
        ...

def das_pinn(net, krnet, pde_residual, bndry, key,
             n_outer=10, n_pinn=10000, n_kr=5000, N_r=2000,
             gamma=1.0, d=10):
    params  = net.init  (key, jnp.zeros(d))
    params_f= krnet.init(key, jnp.zeros(d))
    opt_u   = optax.adam(1e-3); state_u = opt_u.init(params)
    opt_f   = optax.adam(1e-3); state_f = opt_f.init(params_f)

    key, sub = jax.random.split(key)
    S = jax.random.uniform(sub, (N_r, d)) - 0.5

    @jax.jit
    def pinn_step(params, state_u, S):
        loss = lambda p: jnp.mean(jax.vmap(lambda x: pde_residual(p, x))(S)**2) \
                        + gamma * bndry(p)
        g = jax.grad(loss)(params)
        upd, state_u = opt_u.update(g, state_u, params)
        return optax.apply_updates(params, upd), state_u

    @jax.jit
    def kr_step(params_f, state_f, params, S):
        # importance-weighted cross-entropy
        r2 = jax.lax.stop_gradient(
            jax.vmap(lambda x: pde_residual(params, x)**2 * cutoff(x))(S))
        logp_old = jax.lax.stop_gradient(jax.vmap(
            lambda x: krnet.apply(params_f, x, method=krnet.log_prob))(S))
        w  = r2 / (jnp.exp(logp_old) + 1e-12)
        w  = w / jnp.sum(w)
        loss = lambda pf: -jnp.sum(
            w * jax.vmap(lambda x: krnet.apply(pf, x, method=krnet.log_prob))(S))
        g = jax.grad(loss)(params_f)
        upd, state_f = opt_f.update(g, state_f, params_f)
        return optax.apply_updates(params_f, upd), state_f

    for outer in range(n_outer):
        for _ in range(n_pinn):
            params, state_u = pinn_step(params, state_u, S)
        for _ in range(n_kr):
            params_f, state_f = kr_step(params_f, state_f, params, S)
        key, sub = jax.random.split(key)
        z = jax.random.normal(sub, (N_r, d))
        S = jnp.clip(jax.vmap(lambda zz: krnet.inverse(params_f, zz))(z), -0.5, 0.5)
    return params
```

Recommended: KRnet with ~8 coupling blocks, hidden=64, Gaussian prior, logarithmic boundary map (s=2, delta=0.01), 5-10 outer iterations, n_pinn ~ 10^4 inner steps, refresh-or-augment training set.

## Results
On 2-D peaked Poisson, 10-D Fokker-Planck, and high-d Burgers, DAS reduces relative L2 error by 1-2 orders of magnitude vs uniform-LHS PINN at equal collocation budget. Especially effective for low-regularity solutions whose error concentrates in small subregions.

<!-- input quality issue: pymupdf-fallback markdown — equations rendered linearly -->
