import numpy as np
import sys
import os
from numba import njit

# Append the utils directory to the system path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.plotting_utils import (
    plot_defaults,
    plot_losses,
    plot_heatmap_with_circles,
)
from src.losses.loss_functions import weighted_loss_function
from src.algorithms.bho import BeehiveOptimization
from src.algorithms.pso import PSO


plot_defaults()


SAVE = True

if __name__ == "__main__":
    seed = np.random.randint(10000)
    print(f"Running with seed: {seed}")

    # sim properties
    n_particles = 50
    n_iterations = 1000

    # landscape properties and circle properties
    dimension = 2
    radii = [12, 8, 7, 6, 5, 4, 3, 2, 1]
    num_circles = len(radii)
    dim = int(dimension * num_circles)

    @njit()
    def random_seeder(dim, time_steps=10000):
        x = np.random.uniform(0, 1, (dim, dim))
        seed_pos_x = int(np.random.uniform(0, dim))
        seed_pos_y = int(np.random.uniform(0, dim))
        tele_prob = 0.001
        for i in range(time_steps):
            x[seed_pos_x, seed_pos_y] += np.random.uniform(0, 1)
            if np.random.uniform() < tele_prob:
                seed_pos_x = int(np.random.uniform(0, dim))
                seed_pos_y = int(np.random.uniform(0, dim))
            else:
                if np.random.uniform() < 0.5:
                    seed_pos_x += 1
                if np.random.uniform() < 0.5:
                    seed_pos_x += -1
                if np.random.uniform() < 0.5:
                    seed_pos_y += 1
                if np.random.uniform() < 0.5:
                    seed_pos_y += -1
                seed_pos_x = int(max(min(seed_pos_x, dim - 1), 0))
                seed_pos_y = int(max(min(seed_pos_y, dim - 1), 0))
        return x

    weighted_matrix = random_seeder(64, time_steps=100000)
    # weighted_matrix = np.ones((64, 64))

    # -------------------------------------------------------------------------
    # CHOOSE YOUR LOSS FUNCTION HERE
    # -------------------------------------------------------------------------
    # supported loss functions
    LOSS_FUNC = weighted_loss_function  # nD
    low_val = 0
    lower = np.ones(dim) * low_val

    upper = np.ones(dim) * low_val
    print(weighted_matrix.shape)
    high_val_x, high_val_y = weighted_matrix.shape[0], weighted_matrix.shape[1]
    for i in range(dim):
        if i % 2 == 0:
            upper[i] = high_val_x
        else:
            upper[i] = high_val_y

    # -------------------------------------------------------------------------
    # CREATE AND RUN BEEHIVE OPTIMIZER
    # -------------------------------------------------------------------------
    bho = BeehiveOptimization(
        loss_func=LOSS_FUNC,
        weighted_matrix=weighted_matrix,
        radii=radii,
        n_particles=n_particles,
        dim=dim,
        n_iterations=n_iterations,
        rho=0.99,
        c=0.5,
        q=0.1,
        gamma=0.5,
        dt=0.25,
        init_range=10.0,  # only used if bounds=None
        lower_bounds=lower,
        upper_bounds=upper,
        initial_guess=[],
        seed=seed,
    )
    best_pos_bho, best_val_bho = bho.run()

    # It seems like there are different regimes.
    # Firstly we are getting results that are substantially worse now... I don't know
    # exactly. However, this is something that needs to be fixed.
    # Now, conceptually, it seems that we are somehow hitting plateus that need fixing...
    # This could perhaps be done by iteratively changing the 'rho'. Or perhaps,
    # some other parameter that allows the bees to become more or less desperate... i.e.
    # chasing the pheramones less or more.
    # individual values for c and q? what about momentum?
    # i feel like we are not converging the same way anymore...

    print("\nBest solution found (constrained):")
    print("Position:", best_pos_bho)
    print("Loss:", best_val_bho)

    pso = PSO(
        loss_func=LOSS_FUNC,
        n_particles=n_particles,
        weighted_matrix=weighted_matrix,
        radii=radii,
        dim=dim,
        n_iterations=n_iterations,
        w=0.7,  # inertia
        c1=1.5,  # cognitive
        c2=1.5,  # social
        init_range=10.0,
        lower_bounds=lower,
        upper_bounds=upper,
        initial_guess=[],
        seed=seed,
    )

    best_pos_pso, best_val_pso = pso.run()
    print("\nBest solution found (constrained):")
    print("Position:", best_pos_pso)
    print("Loss:", best_val_pso)
    if SAVE:
        dir = "results"
        os.makedirs(dir, exist_ok=True)
        savename_bho = f"{dir}/bho.png"
        savename_pso = f"{dir}/pso.png"
        savename_bho_loss = f"{dir}/bho_loss.png"
        savename_pso_loss = f"{dir}/pso_loss.png"

        plot_heatmap_with_circles(
            weighted_matrix, best_pos_bho, radii, savename=savename_bho
        )
        plot_heatmap_with_circles(
            weighted_matrix, best_pos_pso, radii, savename=savename_pso
        )

        plot_losses(
            bho.history_losses, bho.history_best_loss, savename=savename_bho_loss
        )
        plot_losses(
            pso.history_losses, pso.history_best_loss, savename=savename_pso_loss
        )
