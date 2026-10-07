import numpy as np
from scipy.integrate import solve_ivp

def lorenz63(t, state):
    x, y, z = state

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return [dxdt, dydt, dzdt]

def lorenz96(t, u, K=40, F=8.0, d=1.0):
    du = np.zeros(K)

    for k in range(K):
        km1 = (k - 1) % K
        km2 = (k - 2) % K
        kp1 = (k + 1) % K

        du[k] = (
            u[km1] * (u[kp1] - u[km2])
            - d * u[k]
            + F
        )

    return du

def lorenz96two(t, entries, K=40, F=8.0, d=1.0,
                gamma_j=0.05, ds=None, J=5):

    # If fast-variable damping values are not supplied
    if ds is None:
        ds = np.ones(J)

    # Split flattened state into u and v
    u = entries[:K]
    v = entries[K:].reshape((K, J))

    du = np.zeros(K)
    dv = np.zeros((K, J))

    for k in range(K):

        km1 = (k - 1) % K
        km2 = (k - 2) % K
        kp1 = (k + 1) % K

        # Coupling contribution from fast variables
        sum_term = 0.0

        for j in range(J):
            sum_term += gamma_j * v[k, j] * u[k]

            dv[k, j] = (
                -ds[j] * v[k, j]
                - gamma_j * u[k]**2
            )

        du[k] = (
            u[km1] * (u[kp1] - u[km2])
            + sum_term
            - d * u[k]
            + F
        )

    return np.concatenate((du, dv.flatten()))

def generate_B(model_name, dt=0.1, t_final=1000.0, t_transient=100.0):
    """
    Generate a sample covariance matrix B for one of:
        'lorenz63'
        'lorenz96'
        'lorenz96two'
    """

    model_name = model_name.lower()

    # -------------------------
    # Select model + parameters
    # -------------------------
    if model_name == "lorenz63":
        model = lorenz63
        x0 = np.array([1.0, 1.0, 1.0])

        model_args = ()

    elif model_name == "lorenz96":
        K = 40
        F = 8.0
        d = 1.0

        model = lorenz96

        # Typical Lorenz-96 initial condition
        x0 = F * np.ones(K)
        x0[0] += 0.01

        model_args = (K, F, d)

    elif model_name == "lorenz96two":
        K = 40
        J = 5
        F = 8.0
        d = 1.0
        gamma_j = 0.05
        ds = np.ones(J)

        model = lorenz96two

        # Slow variables
        u0 = F * np.ones(K)
        u0[0] += 0.01

        # Fast variables
        v0 = np.zeros((K, J))

        x0 = np.concatenate((u0, v0.flatten()))

        model_args = (K, F, d, gamma_j, ds, J)

    else:
        raise ValueError(
            "model_name must be 'lorenz63', 'lorenz96', or 'lorenz96two'"
        )

    # -------------------------
    # Simulate model
    # -------------------------
    t_eval = np.arange(0, t_final + dt, dt)

    sol = solve_ivp(
        model,
        [0, t_final],
        x0,
        t_eval=t_eval,
        args=model_args,
        rtol=1e-9,
        atol=1e-11
    )

    trajectory = sol.y.T

    # -------------------------
    # Remove transient
    # -------------------------
    mask = sol.t >= t_transient
    trajectory_cov = trajectory[mask]

    # -------------------------
    # Compute sample mean
    # -------------------------
    mean_state = np.mean(trajectory_cov, axis=0)

    # -------------------------
    # Compute B manually
    # -------------------------
    N = trajectory_cov.shape[0]
    n_state = trajectory_cov.shape[1]

    B = np.zeros((n_state, n_state))

    for k in range(N):
        diff = trajectory_cov[k] - mean_state
        B += np.outer(diff, diff)

    B = B / (N - 1)

    return B
