import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models import lorenz63, lorenz96, lorenz96two, generate_B

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
times = np.arange(T0,TF + DT, DT)
num_steps = len(times)

# Define the measurement covariance matrix

def model_rhs(model_name, t, state):
    if model_name == 'lorenz63':
        return lorenz63(t, state)
    elif model_name == 'lorenz96':
        return lorenz96(t, state, K, F, d)
    elif model_name == 'lorenz96two':
        return lorenz96two(
            t, state, K, F, d, gamma_j, ds, J)
    else:
        raise ValueError("Not a valid model name")

def threeDeeVar(H,R,B,y,mew):
    first = H.T @ np.linalg.inv(R) @ H + np.linalg.inv(B)
    second = H.T @ np.linalg.inv(R) @ y + np.linalg.inv(B) @ mew
    return np.linalg.inv(first) @ second

def get_initial_state(model_name):
    if model_name == 'lorenz63':
        return np.array([1.0, 1.0, 1.0])
    elif model_name == 'lorenz96':
        state = F * np.ones(K)
        state[0] += 0.01
        return state
    elif model_name == 'lorenz96two':
        # K slow states + K*J fast states
        u = F * np.ones(K)
        u[0] += 0.01
        v = np.zeros((K, J))
        return np.concatenate((u, v.flatten()))
    else:
        raise ValueError("Not a valid model name")


def propagate(model_name, t0, t1, state):
    if model_name == 'lorenz63':
        rhs = lambda t, x: lorenz63(t, x)

    elif model_name == 'lorenz96':
        rhs = lambda t, x: lorenz96(t, x, K, F, d)

    elif model_name == 'lorenz96two':
        rhs = lambda t, x: lorenz96two(
            t, x, K, F, d, gamma_j, ds, J
        )
    else:
        raise ValueError("Invalid model name")

    sol = solve_ivp(
        rhs,
        (t0, t1),
        state,
        t_eval=[t1]
    )

    return sol.y[:, -1]

def make_R(H,sigma):
    mean_magnitude = np.mean(np.abs(mean_state))
    sigma = noise_fraction * mean_magnitude

    # Number of observed quantities
    obs_dim = H.shape[0]
    R = sigma * np.eye(obs_dim)
    return R

def make_H(model_name, option="full"):
    if model_name == "lorenz63":
        n = 3
    elif model_name == "lorenz96":
        n = K
    elif model_name == "lorenz96two":
        n = K + K*J
    else:
        raise ValueError("Invalid model name")
    if option == "full":
        return np.eye(n)

    elif option == "first":
        H = np.zeros((1, n))
        H[0, 0] = 1
        return H

    else:
        raise ValueError("Invalid observation option")


def main():
    real_state = get_initial_state(MODEL)
    estimated_state = real_state.copy()

    B = generate_B(MODEL)
    H = make_H(MODEL,H_SPARSITY)
    R = make_R(H,SIGMA)

    print("Model:", MODEL)
    print("State dimension:", len(real_state))
    print("State shape:", real_state.shape)
    print("B shape:", B.shape)
    
    state_dim = len(real_state)

    # Save results over time
    real_hist = np.zeros((num_steps, state_dim))
    estimated_hist = np.zeros((num_steps, state_dim))

    for k in range(1,num_steps):
        t_prev = times[k-1]
        t_now = times[k]

        # Propagate the true state to current time
        real_state = propagate(MODEL,t_prev,t_now,real_state)
        mew = propagate(MODEL,t_prev,t_now,real_state)





    


    return


if __name__ == "__main__":
    main()
