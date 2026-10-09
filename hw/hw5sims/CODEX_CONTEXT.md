# Codex Context: 3DVAR / 4DVAR Homework

## Goal
Extend an existing Python simulation that already runs 3DVAR for:
- `lorenz63`
- `lorenz96`
- `lorenz96two`

The new task is to implement the numerical experiments for one-interval 4DVAR and compare against 3DVAR.

## Existing structure
Typical globals:
```python
MODEL = 'lorenz63'
H_SPARSITY = 'full'

K = 40
J = 5
F = 8.0
d = 1.0
gamma_j = 0.05
ds = np.ones(J)

DT = 0.1
T0 = 0.0
TF = 10.0
times = np.arange(T0, TF + DT, DT)
num_steps = len(times)
```

State dimensions:
- Lorenz-63: 3
- Lorenz-96: 40
- Two-layer Lorenz-96: 240

Existing helpers are similar to:
```python
def model_rhs(model_name, t, state): ...
def get_initial_state(model_name): ...
def propagate(model_name, t0, t1, state): ...
def make_H(model_name, option="full"): ...
def make_R(H, mean_magnitude, noise_fraction): ...
def threeDeeVar(H, R, B, y, mu): ...
```

## Current 3DVAR loop
1. Propagate true state.
2. Generate `y = H @ x_true + noise`, `noise ~ N(0,R)`.
3. Propagate previous analysis to form `mu`.
4. Apply 3DVAR.
5. Save truth and estimate.

## Background covariance
`generate_B(model_name)` runs a long simulation and computes
\[
B=\frac{1}{N-1}\sum_{k=1}^N(x_k-\bar{x})(x_k-\bar{x})^T.
\]

It also returns a mean solution magnitude used in
\[
R=\sigma I,
\]
with `sigma` equal to 1% or 10% of that mean magnitude.

## 3DVAR formula
\[
x_k^a=
(H^TR^{-1}H+B^{-1})^{-1}
(H^TR^{-1}y_k+B^{-1}\mu_k).
\]

Prefer `np.linalg.solve(...)` over explicit inverses where possible.

## Problem 4 result used for Problem 5
One-step 4DVAR cost:
\[
J(x_k)=
\frac12(y_{k+1}-H\mathcal M(x_k))^TR^{-1}(y_{k+1}-H\mathcal M(x_k))
+
\frac12(x_k-\mu_k)^TB^{-1}(x_k-\mu_k).
\]

Gradient:
\[
\nabla J(x_k)=
-D\mathcal M(x_k)^TH^TR^{-1}(y_{k+1}-H\mathcal M(x_k))
+
B^{-1}(x_k-\mu_k).
\]

Continuous adjoint:
\[
-\dot\lambda=Df(x)^T\lambda,
\]
with terminal condition
\[
\lambda(t_{k+1})=
H^TR^{-1}(y_{k+1}-H\mathcal M(x_k)).
\]

Therefore
\[
\nabla J(x_k)=-\lambda(t_k)+B^{-1}(x_k-\mu_k).
\]

## Coding goals
1. Reuse the existing `MODEL` switch.
2. Reuse the same truth simulation, observations, `H`, `R`, and `B`.
3. Add a 4DVAR cost function for `[t_k,t_{k+1}]`.
4. Add `Df(x)` Jacobians for the selected models.
5. Add a backward adjoint solver with `solve_ivp`.
6. Add a function returning cost and gradient.
7. Use `scipy.optimize.minimize`.
8. Save 4DVAR history.
9. Compute RMSE, final RMSE, and runtime.
10. Compare 3DVAR and 4DVAR using identical cases.

Do not rewrite the entire project; add small helper functions.
