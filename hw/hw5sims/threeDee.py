import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

from models import lorenz63, lorenz96, lorenz96two, generate_B
from plotter import plot_results

MODEL = 'lorenz96'
H_SPARSITY = 'full'
SIGMA_PERCENTAGE = 0.10

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


def threeDeeVar(H,R,B,y,mew):
    B = B + 1e-6 * np.eye(B.shape[0])
    first = H.T @ np.linalg.inv(R) @ H + np.linalg.inv(B)
    second = H.T @ np.linalg.inv(R) @ y + np.linalg.inv(B) @ mew
    return np.linalg.solve(first, second)

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

def make_R(H,mean_magnitude,noise_fraction):
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

    elif option == "alternate":
        observed_indices = np.arange(0, n, 2)
        H = np.zeros((len(observed_indices), n))
        for row, state_index in enumerate(observed_indices):
            H[row, state_index] = 1.0
        return H

    else:
        raise ValueError("Invalid observation option")


def main():
    real_state = get_initial_state(MODEL)
    estimated_state = real_state.copy()

    B, mean_magnitude = generate_B(MODEL)
    H = make_H(MODEL,H_SPARSITY)
    R = make_R(H,mean_magnitude,SIGMA_PERCENTAGE)

    print("Model:", MODEL)
    print("State dimension:", len(real_state))
    print("State shape:", real_state.shape)
    print("B shape:", B.shape)
    
    state_dim = len(real_state)

    # Save results over time
    real_hist = np.zeros((num_steps, state_dim))
    estimated_hist = np.zeros((num_steps, state_dim))
    real_hist[0] = real_state
    estimated_hist[0] = estimated_state

    threeDVar_time = 0.0

    for k in range(1,num_steps):
        t_prev = times[k-1]
        t_now = times[k]

        # Propagate the true state to current time
        real_state = propagate(MODEL,t_prev,t_now,real_state)
        mew = propagate(MODEL,t_prev,t_now,estimated_state)
        
        noise = np.random.multivariate_normal(mean=np.zeros(H.shape[0]),cov=R)
        y = H @ real_state + noise
        start = time.perf_counter()
        estimated_state = threeDeeVar(H,R,B,y,mew)
        end = time.perf_counter()

        threeDVar_time += end - start

        real_hist[k] = real_state
        estimated_hist[k] = estimated_state
        
    rsme = plot_results(MODEL,times,real_hist,estimated_hist,K,J)
    print(f"Total 3DVAR computation time: {threeDVar_time:.6f} seconds")
    print(f"Average time per assimilation step: {threeDVar_time / (num_steps - 1):.6e} seconds")

if __name__ == "__main__":
    main()
