import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Assumes the project already provides:
# model_rhs(...)
# propagate(...)
# make_H(...)
# make_R(...)
# generate_B(...)


def jacobian_rhs(model_name, state):
    """
    Return Df(x) for the selected model.

    TODO:
    1. Implement Lorenz-63 Jacobian.
    2. Validate it with a directional derivative test.
    3. Then add Lorenz-96 and lorenz96two.
    """
    raise NotImplementedError


def propagate_with_dense_output(model_name, t0, t1, state):
    """Forward solve over one assimilation interval with dense output."""

    sol = solve_ivp(
        lambda t, x: model_rhs(model_name, t, x),
        (t0, t1),
        state,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.y[:, -1], sol.sol


def fourdvar_objective_and_grad(
    xk,
    model_name,
    t0,
    t1,
    y_next,
    H,
    R,
    B,
    mu,
):
    """
    One-interval 4DVAR cost and gradient.

    J = 1/2 r^T R^-1 r + 1/2 b^T B^-1 b

    r = y_next - H M(xk)
    b = xk - mu

    grad J = -lambda(t0) + B^-1 b
    """

    x_next, forward_solution = propagate_with_dense_output(
        model_name, t0, t1, xk
    )

    r = y_next - H @ x_next
    b = xk - mu

    Rinv_r = np.linalg.solve(R, r)
    Binv_b = np.linalg.solve(B, b)

    J_obs = 0.5 * r @ Rinv_r
    J_bg = 0.5 * b @ Binv_b
    J = J_obs + J_bg

    lambda_T = H.T @ Rinv_r

    def adjoint_rhs(t, lam):
        x_t = forward_solution(t)
        A = jacobian_rhs(model_name, x_t)
        return -A.T @ lam

    adj_sol = solve_ivp(
        adjoint_rhs,
        (t1, t0),
        lambda_T,
        t_eval=[t0],
        rtol=1e-8,
        atol=1e-10,
    )

    if not adj_sol.success:
        raise RuntimeError(adj_sol.message)

    lambda_0 = adj_sol.y[:, -1]
    grad = -lambda_0 + Binv_b

    return float(J), grad


def fourdvar_analysis(
    model_name,
    t0,
    t1,
    y_next,
    H,
    R,
    B,
    mu,
):
    """Run the one-interval 4DVAR optimization."""

    def fun(x):
        J, _ = fourdvar_objective_and_grad(
            x, model_name, t0, t1, y_next, H, R, B, mu
        )
        return J

    def jac(x):
        _, g = fourdvar_objective_and_grad(
            x, model_name, t0, t1, y_next, H, R, B, mu
        )
        return g

    result = minimize(
        fun,
        x0=mu.copy(),
        jac=jac,
        method="L-BFGS-B",
    )

    return result.x, result
