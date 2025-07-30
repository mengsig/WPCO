import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys


# =============================================================================
# Particle Class
# =============================================================================
class Particle:
    """
    Stores a single particle's position, velocity, loss,
    plus its personal-best position and best loss so far.
    """

    def __init__(self, position, velocity, loss=None):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.loss = loss  # current loss
        self.best_position = None  # personal-best position
        self.best_loss = np.inf  # personal-best loss


# =============================================================================
# Particle Swarm Optimization Class
# =============================================================================
class PSO:
    def __init__(
        self,
        loss_func,
        n_particles,
        dim,
        n_iterations,
        w=0.7,  # inertia weight
        c1=1.5,  # cognitive (personal) acceleration
        c2=1.5,  # social (global) acceleration
        init_range=5.0,  # range for random initialization, used if bounds=None
        lower_bounds=None,  # scalar or array of shape (dim,)
        upper_bounds=None,  # scalar or array of shape (dim,)
        initial_guess=[],
        seed=None,
    ):
        """
        Particle Swarm Optimization for a user-specified loss function to MINIMIZE.

        Parameters:
          loss_func     : function f(x) -> float, x in R^dim
          n_particles   : number of particles
          dim           : dimension of the solution space
          n_iterations  : max number of iterations
          w             : inertia weight
          c1, c2        : acceleration constants (cognitive, social)
          init_range    : used if lower_bounds/upper_bounds are None
          lower_bounds  : scalar or array of shape (dim,)
          upper_bounds  : scalar or array of shape (dim,)
          initial_guess : optional array to bias the initial positions
          seed          : optional random seed
        """
        if seed is not None:
            np.random.seed(seed)

        self.loss_func = loss_func
        self.n_particles = n_particles
        self.dim = dim
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.init_range = init_range

        # set up domain
        if lower_bounds is None:
            lower_bounds = -init_range
        if upper_bounds is None:
            upper_bounds = init_range

        self.lower_bounds = (
            np.full(dim, lower_bounds)
            if np.isscalar(lower_bounds)
            else np.array(lower_bounds)
        )
        self.upper_bounds = (
            np.full(dim, upper_bounds)
            if np.isscalar(upper_bounds)
            else np.array(upper_bounds)
        )

        # Create particles
        self.particles = []
        for _ in range(n_particles):
            # random in [0,1], scaled to domain
            pos_u01 = np.random.rand(dim)
            vel_u01 = np.random.uniform(-1, 1, dim) * 0.2
            if len(initial_guess) == 0:
                position = self.lower_bounds + pos_u01 * (
                    self.upper_bounds - self.lower_bounds
                )
            else:
                # For a partial "initial guess"
                position = np.array(initial_guess) + 0.2 * pos_u01 * (
                    self.upper_bounds - self.lower_bounds
                )
            velocity = vel_u01

            p = Particle(position=position, velocity=velocity)
            p.loss = self.loss_func(p.position)
            p.best_position = p.position.copy()
            p.best_loss = p.loss
            self.particles.append(p)

        # track global best
        self.global_best_position = None
        self.global_best_loss = np.inf

        # For logging & potential visualization
        self.history_positions = []  # list of (n_particles x dim) arrays
        self.history_losses = []  # list of shape [n_particles]
        self.history_best_loss = []  # list of best-loss so far
        self.history_best_position = []  # track best solution so far

        # Initialize global best from the initial swarm
        self._update_global_best()

    def _update_global_best(self):
        """
        Checks all particles to update the global best position/loss if there's improvement.
        """
        for p in self.particles:
            if p.loss < self.global_best_loss:
                self.global_best_loss = p.loss
                self.global_best_position = p.position.copy()

    def run(self):
        """
        Runs the PSO for n_iterations,
        storing the history for visualization and printing a progress bar.
        """
        for t in range(self.n_iterations):
            # 1) Update each particle
            for p in self.particles:
                # random coefficients
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # velocity update:
                # v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                cognitive_term = self.c1 * r1 * (p.best_position - p.position)
                social_term = self.c2 * r2 * (self.global_best_position - p.position)
                p.velocity = self.w * p.velocity + cognitive_term + social_term

                # position update
                p.position += p.velocity

                # enforce constraints
                p.position = np.maximum(p.position, self.lower_bounds)
                p.position = np.minimum(p.position, self.upper_bounds)

                # evaluate new loss
                p.loss = self.loss_func(p.position)

                # update personal best
                if p.loss < p.best_loss:
                    p.best_loss = p.loss
                    p.best_position = p.position.copy()

            # 2) Update global best from any improved personal best
            self._update_global_best()

            # 3) Store states for plotting
            positions = np.array([par.position for par in self.particles])
            losses = np.array([par.loss for par in self.particles])
            self.history_positions.append(positions)
            self.history_losses.append(losses)
            self.history_best_loss.append(self.global_best_loss)
            self.history_best_position.append(self.global_best_position.copy())

            # 4) Print progress
            fraction = (t + 1) / self.n_iterations
            bar_length = 30
            filled_length = int(bar_length * fraction)
            bar = "#" * filled_length + "-" * (bar_length - filled_length)
            # Truncate best position if dimension > 5
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

    # -------------------------------------------------------------------------
    # Visualization for 2D
    # -------------------------------------------------------------------------
    def animate_evolution_2d(
        self,
        gif_filename="pso_evolution.gif",
        xlim=(-5, 5),
        ylim=(-5, 5),
        fps=5,
        resolution=100,
    ):
        """
        If dim=2, creates an animated GIF of all iterations, showing:
          - Heatmap of the loss function
          - Particles (blue dots)
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

        # Scatter objects for the swarm and the global best
        scat_particles = ax.scatter([], [], color="blue", s=10, label="Particles")
        scat_gbest = ax.scatter(
            [], [], color="red", s=120, marker="*", label="Global Best"
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("PSO Evolution (2D)")
        ax.legend()

        def init():
            scat_particles.set_offsets(np.empty((0, 2)))
            scat_gbest.set_offsets(np.empty((0, 2)))
            return scat_particles, scat_gbest

        def update(frame):
            positions = self.history_positions[frame]
            losses = self.history_losses[frame]
            # The global best at this frame:
            gbest_pos = self.history_best_position[frame]

            scat_particles.set_offsets(positions)
            scat_gbest.set_offsets([gbest_pos])

            ax.set_title(f"Iteration {frame + 1}/{self.n_iterations}")
            return scat_particles, scat_gbest

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
        Plots a snapshot of the particles and the global best at a given iteration
        overlaid with a heatmap of the loss function. If iteration is None,
        uses the last iteration. This is only for dim=2.
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
        losses = self.history_losses[iteration]
        gbest_pos = self.history_best_position[iteration]

        # Plot the particles
        ax.scatter(
            positions[:, 0], positions[:, 1], color="blue", s=10, label="Particles"
        )
        # Plot global best
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
