---
slot: 71
title: "Wavelets based physics informed neural networks to solve non-linear differential equations"
authors: [Ziya Uddin, Sai Ganga, Rishi Asthana, Wubshet Ibrahim]
year: 2023
venue: Scientific Reports 13
gitrepo: ""
doi: 10.1038/s41598-023-29806-3
---

## TL;DR
Application paper that swaps the PINN activation from `tanh` to one of three **wavelet activations** — Morlet, Mexican-hat, or Gaussian wavelet — and shows improved accuracy on the Blasius boundary-layer ODE, a linear/nonlinear coupled ODE system, and 1-D viscous Burgers. The change is a one-line substitution but the wavelets' compact frequency-space support yields faster convergence and lower L2 error than `tanh` baselines from prior PINN studies.

## Problem
For nonlinear ODE/PDE problems with localized features (Blasius profile, Burgers shock at `t -> 1`), the standard `tanh` MLP needs many neurons and Adam iterations; spectral bias slows convergence around the transition layer. Choosing a better activation is a cheap and orthogonal lever.

## Method
Replace activation `g(.)` in `z_i = g(W_i z_{i-1} + b_i)` with one of:
$$
f_0(x) = \cos(1.75\,x)\,e^{-x^2/2}\quad\text{(Morlet)}
$$
$$
f_1(x) = (1 - x^2)\,e^{-x^2/2}\quad\text{(Mexican hat)}
$$
$$
f_2(x) = -x\,e^{-x^2/2}\quad\text{(Gaussian wavelet)}
$$
Architectures used: 3-5 hidden layers x 20-50 units, Xavier init, full-batch gradient descent. Adam with **decaying learning rate** (start `~1e-2`, decay by 0.5 every few thousand steps). PINN loss is the standard split `L = L_f + L_b` with mean-squared residual and mean-squared BC/IC.

Blasius equation: `2 f''' + f f'' = 0` on `[0, eta_max]`, BCs `f(0) = 0, f'(0) = 0, f'(eta_max) = 1`. Loss:
$$
\mathcal L_f = \frac{1}{N_f}\sum_i (2 f'''(x_i) + f(x_i)\,f''(x_i))^2,\quad
\mathcal L_b = \frac{1}{N_b}\sum_i \big(f(x_0^i)^2 + f'(x_0^i)^2 + (f'(x_1^i) - 1)^2\big)
$$

For Burgers `u_t + u u_x - nu u_xx = 0` with `nu = 0.1` or `0.01/pi`, IC `sin(pi x)`, periodic/zero BC: usual residual loss with collocation points uniformly drawn from `(t, x)`.

For coupled ODE systems, the authors compare a single MLP with multi-head output vs. two independent MLPs in parallel; the parallel choice helps when the two component scales differ.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

def wavelet(x, kind="morlet"):
    if kind == "morlet":
        return jnp.cos(1.75 * x) * jnp.exp(-x**2 / 2)
    if kind == "mexhat":
        return (1.0 - x**2) * jnp.exp(-x**2 / 2)
    if kind == "gauss":
        return -x * jnp.exp(-x**2 / 2)
    raise ValueError(kind)

class WaveletPINN(nn.Module):
    width: int = 30
    depth: int = 4
    out_d: int = 1
    kind:  str = "morlet"

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = wavelet(nn.Dense(self.width, kernel_init=nn.initializers.xavier_uniform())(x),
                        self.kind)
        return nn.Dense(self.out_d)(x)

# Blasius: scalar input eta -> scalar f
def blasius_loss(params, apply_fn, eta_r, eta_0, eta_inf):
    def f(eta):  return apply_fn(params, jnp.atleast_1d(eta)).squeeze()
    fp   = jax.grad(f)
    fpp  = jax.grad(fp)
    fppp = jax.grad(fpp)
    Lf = jnp.mean(jax.vmap(lambda e: (2 * fppp(e) + f(e) * fpp(e))**2)(eta_r.squeeze()))
    f0  = f(eta_0.squeeze()); fp0 = fp(eta_0.squeeze())
    fpi = fp(eta_inf.squeeze())
    Lb  = f0**2 + fp0**2 + (fpi - 1.0)**2
    return Lf + Lb

net    = WaveletPINN(width=30, depth=4, kind="morlet")
params = net.init(jax.random.PRNGKey(0), jnp.zeros((1,)))
sched  = optax.exponential_decay(1e-2, transition_steps=5000, decay_rate=0.5,
                                 staircase=True)
opt    = optax.adam(sched); state = opt.init(params)

@jax.jit
def step(params, state, eta_r, eta_0, eta_inf):
    grads = jax.grad(blasius_loss)(params, net.apply, eta_r, eta_0, eta_inf)
    upd, state = opt.update(grads, state, params)
    return optax.apply_updates(params, upd), state

for it in range(30000):
    eta_r   = jnp.linspace(0.0, 10.0, 200).reshape(-1, 1)
    eta_0   = jnp.zeros((1, 1))
    eta_inf = jnp.full((1, 1), 10.0)
    params, state = step(params, state, eta_r, eta_0, eta_inf)
```

The same template works for the coupled ODEs (multi-output network or parallel networks) and Burgers (input `(t, x)`).

## Results
Across Blasius, linear coupled, nonlinear coupled, and Burgers (`nu = 0.1` and `nu = 0.01/pi`), Morlet/Mexican-hat/Gaussian-wavelet PINNs achieve relative L2 errors `~1e-4..1e-3`, matching or improving on prior `tanh`-PINN, ELM, and X-TFC baselines on the same benchmarks. Morlet usually edges out the others for problems with oscillatory or sharp features; runtime is unchanged (single-line activation change).
