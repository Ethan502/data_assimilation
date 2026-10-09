# 4DVAR Implementation Plan

## 1. Model Jacobian
Add:
```python
def jacobian_rhs(model_name, state):
    ...
```
Return `Df(x)` with shape `(n,n)`.

Start with Lorenz-63 and validate before adding the larger models.

## 2. Forward propagation with dense output
The adjoint needs the forward state along the interval.

```python
def propagate_with_dense_output(model_name, t0, t1, state):
    ...
```

Use:
```python
solve_ivp(..., dense_output=True)
```

Return:
- state at `t1`
- `sol.sol`, so the adjoint can query `x(t)`

## 3. Cost
For candidate `xk`:
```python
r = y_next - H @ x_next
b = xk - mu

J_obs = 0.5 * r @ np.linalg.solve(R, r)
J_bg  = 0.5 * b @ np.linalg.solve(B, b)
J = J_obs + J_bg
```

## 4. Adjoint terminal condition
```python
lambda_T = H.T @ np.linalg.solve(R, r)
```

## 5. Adjoint ODE
\[
\dot\lambda=-Df(x(t))^T\lambda.
\]

When integrating backward:
```python
def adjoint_rhs(t, lam):
    x = forward_solution(t)
    A = jacobian_rhs(model_name, x)
    return -A.T @ lam
```

Use:
```python
solve_ivp(adjoint_rhs, (t1, t0), lambda_T, ...)
```

## 6. Gradient
```python
lambda_0 = adj_sol.y[:, -1]
grad_bg = np.linalg.solve(B, xk - mu)
grad = -lambda_0 + grad_bg
```

## 7. Objective and gradient together
Prefer:
```python
def fourdvar_objective_and_grad(...):
    return J, grad
```

This avoids duplicating mathematical logic.

## 8. Optimizer
Use:
```python
result = minimize(
    fun,
    x0=mu.copy(),
    jac=jac,
    method="L-BFGS-B",
)
```

## 9. Validation
Before running long experiments, verify the analytic/adjoint gradient with a directional derivative:
\[
\frac{J(x+\epsilon h)-J(x-\epsilon h)}{2\epsilon}
\approx \nabla J(x)^T h.
\]

Do this first for Lorenz-63.

## 10. Main loop
At each interval:
1. propagate truth
2. create observation at `t_{k+1}`
3. form background `mu`
4. solve one-interval 4DVAR optimization
5. save estimate
6. later compute RMSE/runtime
