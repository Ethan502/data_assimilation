import numpy as np
import matplotlib.pyplot as plt

def plot_results(model_name, times, real_hist, estimated_hist, K=40, J=5):
    """
    Plot 3DVAR results.

    Parameters
    ----------
    model_name : str
        'lorenz63', 'lorenz96', or 'lorenz96two'

    times : ndarray
        1D array of simulation times.

    real_hist : ndarray
        True state history with shape (num_steps, state_dim).

    estimated_hist : ndarray
        Estimated state history with shape (num_steps, state_dim).

    K : int
        Number of slow variables for Lorenz-96 models.

    J : int
        Number of fast variables per slow variable for two-layer Lorenz-96.

    Returns
    -------
    rmse : ndarray
        RMSE at each time step.
    """

    model_name = model_name.lower()

    # ===================================================
    # 1. RMSE over time
    # ===================================================

    error = estimated_hist - real_hist

    rmse = np.sqrt(
        np.mean(error**2, axis=1)
    )

    plt.figure(figsize=(8, 4))

    plt.plot(times, rmse)

    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.title(f"{model_name} - RMSE over Time")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Final RMSE: {rmse[-1]:.6f}")
    print(f"Mean RMSE:  {np.mean(rmse):.6f}")


    # ===================================================
    # 2. Overall trajectory comparison
    # ===================================================

    if model_name == "lorenz63":

        fig = plt.figure(figsize=(12, 5))

        # True trajectory
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")

        ax1.plot(
            real_hist[:, 0],
            real_hist[:, 1],
            real_hist[:, 2]
        )

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_title("True Lorenz-63 Trajectory")

        # Estimated trajectory
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        ax2.plot(
            estimated_hist[:, 0],
            estimated_hist[:, 1],
            estimated_hist[:, 2]
        )

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("Estimated Lorenz-63 Trajectory")

        plt.tight_layout()
        plt.show()


    elif model_name == "lorenz96":

        fig, axes = plt.subplots(
            1, 2,
            figsize=(13, 5),
            sharey=True
        )

        im1 = axes[0].imshow(
            real_hist.T,
            aspect="auto",
            origin="lower",
            extent=[
                times[0],
                times[-1],
                0,
                real_hist.shape[1] - 1
            ]
        )

        axes[0].set_title("True Lorenz-96")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("State index")

        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(
            estimated_hist.T,
            aspect="auto",
            origin="lower",
            extent=[
                times[0],
                times[-1],
                0,
                estimated_hist.shape[1] - 1
            ]
        )

        axes[1].set_title("Estimated Lorenz-96")
        axes[1].set_xlabel("Time")

        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()


    elif model_name == "lorenz96two":

        # -----------------------------------------------
        # Slow variables
        # -----------------------------------------------

        real_u = real_hist[:, :K]
        estimated_u = estimated_hist[:, :K]

        fig, axes = plt.subplots(
            1, 2,
            figsize=(13, 5),
            sharey=True
        )

        im1 = axes[0].imshow(
            real_u.T,
            aspect="auto",
            origin="lower",
            extent=[
                times[0],
                times[-1],
                0,
                K - 1
            ]
        )

        axes[0].set_title("True Two-Layer L96: Slow Variables")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Slow state index")

        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(
            estimated_u.T,
            aspect="auto",
            origin="lower",
            extent=[
                times[0],
                times[-1],
                0,
                K - 1
            ]
        )

        axes[1].set_title("Estimated Two-Layer L96: Slow Variables")
        axes[1].set_xlabel("Time")

        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()


        # -----------------------------------------------
        # Fast variables
        # -----------------------------------------------

        real_v = real_hist[:, K:]
        estimated_v = estimated_hist[:, K:]

        fig, axes = plt.subplots(
            1, 2,
            figsize=(13, 6),
            sharey=True
        )

        im1 = axes[0].imshow(
            real_v.T,
            aspect="auto",
            origin="lower",
            extent=[
                times[0],
                times[-1],
                0,
                real_v.shape[1] - 1
            ]
        )

        axes[0].set_title("True Two-Layer L96: Fast Variables")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Fast state index")

        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(
            estimated_v.T,
            aspect="auto",
            origin="lower",
            extent=[
                times[0],
                times[-1],
                0,
                estimated_v.shape[1] - 1
            ]
        )

        axes[1].set_title("Estimated Two-Layer L96: Fast Variables")
        axes[1].set_xlabel("Time")

        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()


    else:
        raise ValueError(
            "model_name must be "
            "'lorenz63', 'lorenz96', or 'lorenz96two'"
        )


    # ===================================================
    # 3. Individual state-component comparisons
    # ===================================================

    state_dim = real_hist.shape[1]
    states_per_figure = 12

    for start in range(0, state_dim, states_per_figure):

        end = min(
            start + states_per_figure,
            state_dim
        )

        num_states = end - start

        # Dynamically size subplot grid
        if num_states == 1:
            nrows, ncols = 1, 1

        elif num_states <= 3:
            nrows, ncols = 1, num_states

        elif num_states <= 6:
            nrows, ncols = 2, 3

        elif num_states <= 8:
            nrows, ncols = 2, 4

        else:
            nrows, ncols = 3, 4

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 3 * nrows),
            sharex=True
        )

        axes = np.atleast_1d(axes).flatten()

        for i in range(num_states):

            state_index = start + i

            axes[i].plot(
                times,
                real_hist[:, state_index],
                label="True"
            )

            axes[i].plot(
                times,
                estimated_hist[:, state_index],
                label="Estimated",
                linestyle="--"
            )

            # State names
            if model_name == "lorenz63":
                state_names = ["x", "y", "z"]
                title = state_names[state_index]

            elif model_name == "lorenz96":
                title = f"x[{state_index}]"

            elif model_name == "lorenz96two":

                if state_index < K:
                    title = f"u[{state_index}]"

                else:
                    fast_index = state_index - K

                    k_index = fast_index // J
                    j_index = fast_index % J

                    title = f"v[{k_index},{j_index}]"

            axes[i].set_title(title)
            axes[i].grid(True)

            if i == 0:
                axes[i].legend()

        # Hide unused subplot locations
        for i in range(num_states, len(axes)):
            axes[i].axis("off")

        if model_name == "lorenz63":
            fig.suptitle(
                "Lorenz-63: True vs Estimated States"
            )
        else:
            fig.suptitle(
                f"{model_name}: States {start} to {end - 1}"
            )

        fig.supxlabel("Time")
        fig.supylabel("State value")

        plt.tight_layout()
        plt.show()


    return rmse
