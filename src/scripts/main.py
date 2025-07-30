import numpy as np
import sys
import os

# Append the utils directory to the system path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.plotting_utils import plot_defaults, plot_losses
from src.losses.loss_functions import rosenbrock_nd
from src.algorithms.bho import BeehiveOptimization
from src.algorithms.pso import PSO


plot_defaults()


SAVE = False

if __name__ == "__main__":
    seed = np.random.randint(10000)
    print(f"Running with seed: {seed}")

    dim = 10
    n_particles = 15
    n_iterations = 1000

    # -------------------------------------------------------------------------
    # CHOOSE YOUR LOSS FUNCTION HERE
    # -------------------------------------------------------------------------
    # supported loss functions
    LOSS_FUNC = rosenbrock_nd  # nD
    low_val, up_val = -2, 2

    lower = np.ones(dim) * low_val
    upper = np.ones(dim) * up_val

    # -------------------------------------------------------------------------
    # CREATE AND RUN BEEHIVE OPTIMIZER
    # -------------------------------------------------------------------------
    bho = BeehiveOptimization(
        loss_func=LOSS_FUNC,
        n_particles=n_particles,
        dim=dim,
        n_iterations=n_iterations,
        rho=None,
        c=1,
        q=0.05,
        gamma=0.5,
        init_range=10.0,  # only used if bounds=None
        lower_bounds=lower,
        upper_bounds=upper,
        initial_guess=[],
        seed=seed,
    )
    best_pos, best_val = bho.run()

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
    print("Position:", best_pos)
    print("Loss:", best_val)

    pso = PSO(
        loss_func=LOSS_FUNC,
        n_particles=n_particles,
        dim=dim,
        n_iterations=n_iterations,
        w=0.7,  # inertia
        c1=1.5,  # cognitive
        c2=1.5,  # social
        init_range=10.0,
        lower_bounds=low_val,
        upper_bounds=up_val,
        initial_guess=[],
        seed=seed,
    )

    best_pos, best_val = pso.run()
    print("\nBest solution found (constrained):")
    print("Position:", best_pos)
    print("Loss:", best_val)
    if SAVE:
        dir = "images"
        os.makedirs(dir, exist_ok=True)
        savename_bho = f"{dir}/bho.png"
        savename_pso = f"{dir}/pso.png"

        plot_losses(bho.history_losses, bho.history_best_loss, savename=savename_bho)
        plot_losses(pso.history_losses, pso.history_best_loss, savename=savename_pso)
