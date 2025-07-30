import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os

# Append the utils directory to the system path for importing PlotDefaults, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

from PlotDefaults import PlotDefaults
from LossFunctions import (
    rosenbrock_nd,
    beale_2d,
    keane_bump_nd,
    shekel_nd,
    rastrigin_nd,
    goldstein_price,
    ackley_nd,
    griewank_nd,
    sphere_nd,
    zakharov_nd,
    ellipsoid_nd,
)


class GeneticAlgorithm:
    """
    A simple Genetic Algorithm for MINIMIZING a user-defined loss function.
    """

    def __init__(
        self,
        loss_func,
        pop_size,
        dim,
        n_iterations,
        lower_bounds,
        upper_bounds,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism_ratio=0.1,
        init_range=5.0,
        initial_guess=[],
        seed=None,
    ):
        """
        Args:
          loss_func      : function f(x) -> float, to be minimized
          pop_size       : number of individuals in the population
          dim            : dimension of the solution space
          n_iterations   : number of generations
          lower_bounds   : scalar or array for lower domain limit
          upper_bounds   : scalar or array for upper domain limit
          crossover_rate : probability of performing crossover
          mutation_rate  : probability of mutating each gene
          elitism_ratio  : fraction of top individuals carried over each gen
          init_range     : used if lower_bounds/upper_bounds are None
          initial_guess  : optional array to bias initial positions
          seed           : optional random seed
        """
        if seed is not None:
            np.random.seed(seed)

        self.loss_func = loss_func
        self.pop_size = pop_size
        self.dim = dim
        self.n_iterations = n_iterations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio
        self.init_range = init_range

        # Convert bounds to arrays if needed
        if np.isscalar(lower_bounds):
            self.lower_bounds = np.full(dim, lower_bounds)
        else:
            self.lower_bounds = np.array(lower_bounds)

        if np.isscalar(upper_bounds):
            self.upper_bounds = np.full(dim, upper_bounds)
        else:
            self.upper_bounds = np.array(upper_bounds)

        # Initialize population
        self.population = []
        for _ in range(pop_size):
            pos_u01 = np.random.rand(dim)
            if len(initial_guess) == 0:
                position = self.lower_bounds + pos_u01 * (
                    self.upper_bounds - self.lower_bounds
                )
            else:
                # partial "initial guess"
                position = np.array(initial_guess) + 0.2 * pos_u01 * (
                    self.upper_bounds - self.lower_bounds
                )
            self.population.append(position)
        self.population = np.array(self.population, dtype=float)

        # Evaluate losses of initial population
        self.losses = np.array([self.loss_func(ind) for ind in self.population])

        # Track the global best
        best_idx = np.argmin(self.losses)
        self.global_best_position = self.population[best_idx].copy()
        self.global_best_loss = self.losses[best_idx]

        # Logging for visualization
        self.history_positions = []
        self.history_losses = []
        self.history_best_loss = []
        self.history_best_position = []

        # Store initial iteration
        self._store_history()

    def _store_history(self):
        """
        Logs the current population state for potential plotting.
        """
        self.history_positions.append(self.population.copy())
        self.history_losses.append(self.losses.copy())
        self.history_best_loss.append(self.global_best_loss)
        self.history_best_position.append(self.global_best_position.copy())

    def run(self):
        """
        Run the GA for n_iterations generations,
        storing the best solution and printing a progress bar.
        """
        for t in range(self.n_iterations):
            # 1) Selection (roulette wheel or rank-based).
            #    For simplicity, let's do a 'tournament' or 'roulette' approach.
            new_population = []

            # We'll use elitism: keep the top fraction of individuals
            n_elites = int(self.elitism_ratio * self.pop_size)
            sorted_indices = np.argsort(self.losses)  # ascending (best = first)
            elites = self.population[sorted_indices[:n_elites]]
            new_population.extend(elites)

            # For the rest, we do pairs of parents from the population,
            # then possibly crossover, then mutate.
            while len(new_population) < self.pop_size:
                parent1 = self._select_parent()
                parent2 = self._select_parent()

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)

            # convert new_population to array
            self.population = np.array(new_population[: self.pop_size], dtype=float)

            # 2) Enforce domain constraints
            self.population = np.maximum(self.population, self.lower_bounds)
            self.population = np.minimum(self.population, self.upper_bounds)

            # 3) Recompute losses
            self.losses = np.array([self.loss_func(ind) for ind in self.population])

            # 4) Update global best
            best_idx = np.argmin(self.losses)
            if self.losses[best_idx] < self.global_best_loss:
                self.global_best_loss = self.losses[best_idx]
                self.global_best_position = self.population[best_idx].copy()

            # 5) Log history
            self._store_history()

            # 6) Print progress bar
            fraction = (t + 1) / self.n_iterations
            bar_length = 30
            filled_length = int(bar_length * fraction)
            bar = "#" * filled_length + "-" * (bar_length - filled_length)

            if self.dim <= 5:
                best_pos_str = np.round(self.global_best_position, 4)
            else:
                short_vec = np.round(self.global_best_position[:3], 4)
                best_pos_str = f"{short_vec}..."

            msg = (
                f"\rIteration {t + 1}/{self.n_iterations} [{bar}] "
                f"Best Loss: {self.global_best_loss:.4f} "
                f"Best Pos: {best_pos_str} "
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

        sys.stdout.write("\n")
        return self.global_best_position, self.global_best_loss

    def _select_parent(self):
        """
        Parent selection method.
        We'll do a simple "roulette wheel" selection based on 1 / loss.
        Alternatively, you could do tournament selection or rank selection.
        """
        # Because we want to MINIMIZE the loss, let's invert the loss to measure "fitness".
        # But handle cases when the loss might be zero or extremely small:
        fitness = 1.0 / (self.losses + 1e-12)
        total_fitness = np.sum(fitness)
        pick = np.random.rand() * total_fitness

        cum_sum = 0.0
        for i in range(self.pop_size):
            cum_sum += fitness[i]
            if cum_sum >= pick:
                return self.population[i].copy()
        # fallback (unlikely due to floating inaccuracies)
        return self.population[-1].copy()

    def _crossover(self, parent1, parent2):
        """
        Single-point or uniform crossover if we pass the probability threshold.
        """
        child1, child2 = parent1.copy(), parent2.copy()
        if np.random.rand() < self.crossover_rate:
            # Let's do a single-point crossover
            point = np.random.randint(1, self.dim)  # in [1, dim-1]
            child1[:point], child2[:point] = parent2[:point], parent1[:point]
        return child1, child2

    def _mutate(self, individual):
        """
        Mutates each gene with probability self.mutation_rate.
        The mutation is a random small shift within the domain.
        """
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                # shift with some fraction of domain size
                range_size = self.upper_bounds[i] - self.lower_bounds[i]
                mutation_amount = 0.1 * range_size * (2 * np.random.rand() - 1)
                individual[i] += mutation_amount
        return individual

    # -------------------------------------------------------------------------
    # 2D Visualization
    # -------------------------------------------------------------------------
    def animate_evolution_2d(
        self,
        gif_filename="ga_evolution.gif",
        xlim=(-5, 5),
        ylim=(-5, 5),
        fps=5,
        resolution=100,
    ):
        """
        If dim=2, creates an animated GIF of all iterations, showing:
          - Heatmap of the loss function
          - Population (blue dots)
          - Global best (red star)
        For dim>2, prints a message and returns without visualization.
        """
        if self.dim != 2:
            print(f"Animation only supported for dim=2. (dim={self.dim})")
            return

        # Create mesh for the chosen loss function
        x_vals = np.linspace(xlim[0], xlim[1], resolution)
        y_vals = np.linspace(ylim[0], ylim[1], resolution)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z_grid = np.zeros_like(X_grid)
        for i in range(resolution):
            for j in range(resolution):
                Z_grid[i, j] = self.loss_func([X_grid[i, j], Y_grid[i, j]])

        fig, ax = plt.subplots(figsize=(8, 7))
        heatmap = ax.pcolormesh(
            X_grid, Y_grid, Z_grid, shading="auto", cmap="hot", norm="log"
        )
        fig.colorbar(heatmap, ax=ax, label="Loss")

        # Scatter objects for the population and the global best
        scat_population = ax.scatter([], [], color="blue", s=10, label="Population")
        scat_gbest = ax.scatter(
            [], [], color="red", s=120, marker="*", label="Global Best"
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Genetic Algorithm Evolution (2D)")
        ax.legend()

        def init():
            scat_population.set_offsets(np.empty((0, 2)))
            scat_gbest.set_offsets(np.empty((0, 2)))
            return scat_population, scat_gbest

        def update(frame):
            positions = self.history_positions[frame]
            # The global best at this frame:
            gbest_pos = self.history_best_position[frame]

            scat_population.set_offsets(positions)
            scat_gbest.set_offsets([gbest_pos])

            ax.set_title(f"Iteration {frame + 1}/{self.n_iterations}")
            return scat_population, scat_gbest

        n_frames = len(self.history_positions)
        anim = FuncAnimation(
            fig, update, frames=n_frames, init_func=init, blit=True, interval=200
        )

        writer = PillowWriter(fps=fps)
        anim.save(gif_filename, writer=writer)
        plt.close(fig)
        print(f"Animation saved to {gif_filename}")

    def visualize_snapshot_2d(
        self, iteration=None, xlim=(-5, 5), ylim=(-5, 5), resolution=100
    ):
        """
        Plots a snapshot of the population and the global best at a given iteration
        overlaid with a heatmap of the loss function. If iteration is None,
        uses the last iteration. Only valid for dim=2.
        """
        if self.dim != 2:
            print(f"visualize_snapshot_2d only works for dim=2. (dim={self.dim})")
            return

        if iteration is None:
            iteration = len(self.history_positions) - 1

        # Create mesh for the chosen loss function
        x_vals = np.linspace(xlim[0], xlim[1], resolution)
        y_vals = np.linspace(ylim[0], ylim[1], resolution)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z_grid = np.zeros_like(X_grid)
        for i in range(resolution):
            for j in range(resolution):
                Z_grid[i, j] = self.loss_func([X_grid[i, j], Y_grid[i, j]])

        fig, ax = plt.subplots(figsize=(8, 7))
        heatmap = ax.pcolormesh(
            X_grid, Y_grid, Z_grid, shading="auto", cmap="hot", norm="log"
        )
        fig.colorbar(heatmap, ax=ax, label="Loss")

        positions = self.history_positions[iteration]
        gbest_pos = self.history_best_position[iteration]

        ax.scatter(
            positions[:, 0], positions[:, 1], color="blue", s=10, label="Population"
        )
        ax.scatter(
            gbest_pos[0],
            gbest_pos[1],
            color="red",
            s=120,
            marker="*",
            label="Global Best",
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Snapshot at Iteration {iteration + 1}/{self.n_iterations}")
        ax.legend()
        plt.show()
