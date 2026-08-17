---
slot: 125
title: "Robust Physics-Informed Penalty-Free Machine Learning Method for Solving 3-D Maxwell's Equations in the Frequency Domain"
authors: [Runwei Zhou, Dan Jiao]
year: 2026
venue: IEEE Transactions on Microwave Theory and Techniques
gitrepo: ""
doi: 10.1109/TMTT.2026.3658422
---

## TL;DR
For 3-D frequency-domain Maxwell PDEs/IEs, discretize the variational form into `Ax = b` (FEM or IE assembly), impose Dirichlet BCs by directly modifying rows/cols of A (zero row/col with diagonal=1) and b — then train a neural network whose output `x(θ)` minimises `L = ½‖A_new x(θ) − b_new‖²`. The loss is unconstrained, quadratic, strongly convex (A^H A is positive definite), so gradient descent has guaranteed convergence. Avoids weighted multi-term PINN losses entirely.

## Problem
Standard PINN loss `L = w₁ L_PDE + w₂ L_BC + w₃ L_IC` is nonlinear, nonconvex, and especially fragile for complex-valued, ill-conditioned 3-D EM problems with mixed Dirichlet/Neumann/Robin BCs, PEC interfaces, and delta-function current sources. Penalty terms balance poorly; AD fails near singularities; the deep Ritz functional is real-valued and incompatible with complex EM.

## Method

### A. Penalty-free formulation
Discretise the variational functional `δF(E) = 0` (which already satisfies Neumann/Robin naturally) via FEM/IE into `A x = b`. For Dirichlet rows i: zero row i and column i of A except `A_ii = 1`; set `b_i` to prescribed value — call this `A_new x = b_new`. Minimise:
$$L_{\text{new}}(\theta) = \tfrac12 (A_{\text{new}}x(\theta) - b_{\text{new}})^H (A_{\text{new}}x(\theta) - b_{\text{new}})$$
Because `A_new^H A_new` is positive definite, gradient descent converges for any step `0 < α < 2/λ_max(A_new^H A_new)`.

### B. Gradient via Jacobian of NN output
$$\nabla_\theta L = (\nabla_\theta x)^H A_{\text{new}}^H (A_{\text{new}} x^{(i)} - b_{\text{new}})$$
Gradient is directly proportional to residual `(A x − b)`, unlike standard PINN losses where the gradient bears no correlation with accuracy.

### C. NN architecture: BN + Residual + Multitask
For each field component (Ex, Ey, Ez, real & imag) a separate "task head" sharing input `(x,y,z)`. Each task is a stack of residual blocks; each block = (FC + BN + FC + BN) with a skip connection. BN crucial because physics-driven outputs span huge dynamic ranges.

### D. Conditioning fix
For electrically small problems the curl-curl `k₀² ε_r E` term becomes negligibly small relative to `∇×(μ_r⁻¹ ∇×E)`, blowing up condition number. Use a well-conditioned reformulation (e.g. tree-cotree splitting or augmented VIE) before assembly. Similarly for VIE, use Eqs. (24)-(26) with auxiliary scalar potential.

### E. H²-acceleration for dense IE
Replace dense MVM `Ax` by hierarchical `A = Σ_l U_l Λ_l V_l + A_ina` with linear complexity, evaluated each forward pass.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

class ResBlock(nn.Module):
    width: int
    @nn.compact
    def __call__(self, x, train: bool):
        h = nn.BatchNorm(use_running_average=not train)(nn.Dense(self.width)(x))
        h = jax.nn.relu(h)
        h = nn.BatchNorm(use_running_average=not train)(nn.Dense(self.width)(h))
        return jax.nn.relu(x + h)

class TaskHead(nn.Module):
    width: int = 300
    depth: int = 5
    n_unknowns: int = 100
    @nn.compact
    def __call__(self, h, train: bool):
        for _ in range(self.depth):
            h = ResBlock(self.width)(h, train)
        return nn.Dense(2 * self.n_unknowns)(h)               # real, imag

class MaxwellNet(nn.Module):
    n_tasks: int = 6
    width: int = 300
    depth: int = 5
    n_unknowns_per_task: int = 100
    @nn.compact
    def __call__(self, xyz, train: bool):
        h = jax.nn.relu(nn.Dense(self.width)(xyz))
        outs = []
        for k in range(self.n_tasks):
            o = TaskHead(self.width, self.depth, self.n_unknowns_per_task,
                         name=f"head_{k}")(h, train)
            half = o.shape[-1] // 2
            outs.append(o[..., :half] + 1j * o[..., half:])
        return jnp.concatenate(outs, axis=-1)                 # (N_grid, N_unk) complex

def penalty_free_loss(params, batch_stats, xyz, A_new, b_new):
    y, new_bs = net.apply({"params": params, "batch_stats": batch_stats},
                          xyz, train=True, mutable=["batch_stats"])
    x = y.mean(axis=0)                                        # (N_unk,) complex
    r = A_new @ x - b_new
    return 0.5 * jnp.real(jnp.vdot(r, r)), new_bs

net = MaxwellNet()
vars_ = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)), train=False)
params, batch_stats = vars_["params"], vars_["batch_stats"]
opt = optax.adam(1e-6); opt_state = opt.init(params)

@jax.jit
def step(params, batch_stats, opt_state, xyz, A_new, b_new):
    (loss, new_bs), grads = jax.value_and_grad(penalty_free_loss, has_aux=True)(
        params, batch_stats, xyz, A_new, b_new)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(params, updates), new_bs["batch_stats"], opt_state, loss
```

Hyperparameters: 5 residual blocks, width 100-800, ReLU, multitask = one head per scalar+vector field component (Ex/Ey/Ez/φ × real/imag). Adam, lr 1e-6 to 1e-4 (problem-dependent). FEM unknowns 2728 (fin waveguide) - 134241 (VIE 1 GHz sphere). H²-leaf size 20, η=0.1, p=3 interpolation points.

## Results
Fin waveguide (3 mm cube with PEC fins, 10 GHz, 2728 unknowns): training 306 s, inference 1.1 ms, relative error <2% vs FEM. Cavity-backed patch antenna (5 GHz): error <1% throughout, input impedance matches reference. 2-layer lossy sphere (1 GHz, 134k unknowns): error <1% on E-field. Magnetic meta-surfaces (1 MHz IE): <0.1% error in 628 s. H²-compression: linear memory scaling for capacitance extraction up to 32x32 crossbars.
