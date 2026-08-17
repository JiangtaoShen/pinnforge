---
slot: 111
title: "Diversity-Aware Adaptive Collocation for Physics-Informed Neural Networks via Sparse QUBO Optimization and Hybrid Coresets"
authors: [Hadi Salloum, Maximilian Mifsud Bonici, Sinan Ibrahim, Pavel Osinenko, Alexei Kornaev]
year: 2026
venue: arXiv:2603.06761
gitrepo: ""
---

## TL;DR
Cast PINN collocation point selection as a fixed-budget coreset problem: solve a QUBO/BQM that rewards squared-residual importance and penalises pairwise space-time similarity. To avoid dense all-to-all `k-hot` couplers, drop the cardinality penalty and solve a sparse BQM on a kNN graph, then run an O(K)-marginal repair to enforce the exact `K`-point budget, with stratified "coverage anchors" to keep the PDE globally enforced.

## Problem
Uniform PINN collocation oversamples smooth regions; residual-adaptive refinement (RAR, RAD, R3) tends to cluster correlated points in localized features (shocks, boundary layers), losing global PDE coverage and increasing redundancy. We need a principled subset-selection rule that balances *informativeness* and *diversity* without the O(M²) blow-up of a full pairwise QUBO.

## Method
From a candidate pool `C={(x_i,t_i)}_{i=1..N}` (uniform), warm-start train a short PINN to get `θ_0`. Score each candidate by squared residual `s_i = r_{θ_0}(x_i,t_i)²` (min-max normalised, clipped at 99th percentile). Prefilter to working set of size `M≪N`: top-`βM` by score + uniform `(1-β)M`.

Anisotropic space-time RBF similarity
$$
w_{ij}=\exp\!\Big(-\Big(\tfrac{x_i-x_j}{\ell_x}\Big)^2-\Big(\tfrac{t_i-t_j}{\ell_t}\Big)^2\Big),
$$
kept only on edges of a kNN graph (`|E|=O(Mk)`).

### A. Dense k-hot QUBO baseline
$$
\min_{z\in\{0,1\}^M}\;\sum_i(-\alpha s_i)z_i+\gamma\!\sum_{i<j}w_{ij}z_iz_j+\lambda\Big(\sum_i z_i-K\Big)^2
$$
gives all-to-all couplers; expensive.

### B. Sparse "soft-K" BQM (proposed)
$$
\min_{z\in\{0,1\}^M}\;\sum_i\big(-\alpha s_i+\mu\big)z_i+\gamma\!\!\sum_{(i,j)\in E}\!\!w_{ij}z_iz_j,
$$
with bias `μ ≈ s̄ − K/M`. Solved by simulated annealing (D-Wave Ocean's neal sampler).

### C. Exact-K repair on sparse graph
For an initial set `\hat S`, define marginal utility of a selected point and gain of an unselected point:
$$
U(i\mid S)=\alpha s_i-\gamma\!\!\!\sum_{j\in S\setminus\{i\}}\!\!\!w_{ij},\quad
G(i\mid S)=\alpha s_i-\gamma\!\!\!\sum_{j\in S}\!\!w_{ij}.
$$
If `|\hat S|>K` drop the lowest-utility points; if `<K` add the highest-gain candidates. Only touches kNN-edge neighbours, so O((K+M)·k).

### D. Hybrid coverage anchors
Reserve `K_a=(1-ρ)K` stratified uniform anchor points; QUBO selects only the remaining `ρK`. Prevents over-concentration on shocks.

```python
import jax, jax.numpy as jnp
import numpy as np
import neal
from sklearn.neighbors import NearestNeighbors

def collocation_select(params, apply_fn, pde_residual, pool_xt,
                       M=4000, K=1000,
                       ell_x=0.05, ell_t=0.05, k_nn=8,
                       alpha=1.0, gamma=2.0, mu=None,
                       rho=0.8):
    r = pde_residual(params, apply_fn, pool_xt)
    s = np.asarray(jnp.square(r))                       # squared residual
    s = np.clip(s / np.quantile(s, 0.99), 0, 1)
    top = np.argsort(-s)[:M//2]
    rnd = np.random.choice(len(s), M//2, replace=False)
    idx = np.unique(np.concatenate([top, rnd]))
    pts = np.asarray(pool_xt)[idx]; s = s[idx]
    Xs = np.stack([pts[:, 0]/ell_x, pts[:, 1]/ell_t], 1)
    nbr = NearestNeighbors(n_neighbors=k_nn+1).fit(Xs)
    dist, neigh = nbr.kneighbors(Xs)
    edges = [(i, j, np.exp(-((Xs[i]-Xs[j])**2).sum()))
             for i in range(len(idx)) for j in neigh[i][1:]]
    K_q = int(rho * K); K_a = K - K_q
    bqm = {(i, i): -alpha*s[i] + (mu or (s.mean() - K_q/len(idx)))
           for i in range(len(idx))}
    for i, j, w in edges:
        bqm[(min(i, j), max(i, j))] = bqm.get((min(i, j), max(i, j)), 0.0) + gamma*w
    sampler = neal.SimulatedAnnealingSampler()
    res = sampler.sample_qubo(bqm, num_reads=20).first.sample
    sel = [i for i, v in res.items() if v == 1]
    while len(sel) > K_q:                               # drop lowest-utility
        U = [(alpha*s[i] - gamma*sum(w for (a, b, w) in edges
              if i in (a, b) and (b if a == i else a) in sel), i)
             for i in sel]
        sel.remove(min(U)[1])
    while len(sel) < K_q:
        cand = set(range(len(idx))) - set(sel)
        G = [(alpha*s[i] - gamma*sum(w for (a, b, w) in edges
              if i in (a, b) and (b if a == i else a) in sel), i)
             for i in cand]
        sel.append(max(G)[1])
    anchors = stratified_uniform_sample(pool_xt, K_a)
    return jnp.concatenate([jnp.asarray(np.asarray(pool_xt)[idx[sel]]), anchors])
```

Hyper-parameters: `M≈2000-5000`, prefilter mix `β≈0.5`, `k_nn≈8`, `ℓ_x≈0.05·L_x`, `ℓ_t≈0.05·L_t`, `α=1`, `γ∈[1,5]`, hybrid fraction `ρ≈0.8`, refresh every 1-2k Adam steps.

## Results
On 1-D viscous Burgers with shock formation (`ν=0.01/π`), the sparse-BQM + repair + hybrid anchors matches dense-QUBO accuracy at `~10×` lower selection wall-clock than dense formulations, beats RAR-D / RAD baselines in time-to-accuracy at fixed point budget `K`, and avoids the residual-redundancy clustering of pure residual sampling.
