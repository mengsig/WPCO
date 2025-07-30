import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys


# =============================================================================
# Bee and Pheromone Classes
# =============================================================================
class Bee:
    """
    Stores a single bee's position, velocity, and loss.
    """

    def __init__(self, position, velocity, loss=None):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.loss = loss  # L(t)
        self.prev_loss = None  # L(t-1)
        self.c = np.random.normal(0.9, 0.1)
        self.q = np.random.normal(0.4, 0.1)


class Pheromone:
    """
    Stores the position and strength of a pheromone.
    """

    def __init__(self, position, strength):
        self.position = np.array(position, dtype=float)
        self.strength = strength


# =============================================================================
# Deterministic Beehive Optimization Class (with constraints)
# =============================================================================
class BeehiveOptimization:
    def __init__(
        self,
        loss_func,
        n_particles,
        dim,
        n_iterations,
        dt=1.0,
        rho=None,  # Inertia
        c=0.9,  # Collective (pheromone) influence
        q=0.1,  # Queen attraction
        w=0.0,  # Wasp danger
        gamma=0.5,  # Pheromone decay factor
        kappa=None,  # Initial velocity scaling factor.
        init_range=5.0,  # Range for random initialization, e.g. [-init_range, init_range].
        lower_bounds=None,  # Lower bound(s) for constrained optimization
        upper_bounds=None,  # Upper bound(s) for constrained optimization
        initial_guess=[],
        seed=None,
    ):
        """
        Parameters:
          loss_func    : function to MINIMIZE (expects np.array of shape (dim,)).
          n_particles       : number of bees.
          dim          : dimension of the solution space.
          n_iterations : number of iterations.
          dt           : time-step for position updates.
          rho          : inertia factor (0 < rho < 1).
          c            : pheromone influence weight (0 <= c <= 1).
          q            : queen attraction weight (0 <= q <= 1).
          gamma        : pheromone decay rate (0 < gamma <= 1).
          kappa        : starting velocity bounds.
          init_range   : used if lower_bounds/upper_bounds not provided.
          lower_bounds : scalar or array of shape (dim,). If None, uses -init_range.
          upper_bounds : scalar or array of shape (dim,). If None, uses +init_range.
        """
        if seed is not None:
            np.random.seed(seed)
        self.loss_func = loss_func
        self.n_particles = n_particles
        self.dim = dim
        self.n_iterations = n_iterations
        if dt != 1:
            print(f"The user has forced dt = {dt}. It is recommended to keep dt = 1")
        self.dt = dt
        if rho == None:
            self.rho = (0.9**100) ** (1 / self.n_iterations)
            if self.rho >= 0.9:
                print(f"Automated momentum parameter has been set to: rho = {self.rho}")
        else:
            self.rho = rho
        if self.rho < 0.9:
            print(
                f"The momentum parameter rho cannot be set such that rho < 0.9. The parameter rho has automatically been set to 0.9."
            )
            self.rho = 0.9
        self.c = c
        self.q = q
        self.w = w
        self.gamma = gamma
        if kappa == None:
            self.kappa = (np.array(upper_bounds) - np.array(lower_bounds)).mean() / 5
            print(
                f"Automated initial velocity bounds have been set to [{-self.kappa}, {self.kappa}]"
            )
        else:
            self.kappa = kappa
        self.globalIndex = 1
        self.initial_guess = initial_guess

        if c + q > 1:
            print("Warning: c + q > 1. The paper often uses c + q <= 1 for stability.")

        # If user doesn't provide explicit bounds, set them to [-init_range, init_range]
        if lower_bounds is None:
            lower_bounds = -init_range
        if upper_bounds is None:
            upper_bounds = init_range

        # Convert to arrays if needed
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

        # Initialize bees within [lower_bounds, upper_bounds]
        self.bees = []
        for _ in range(n_particles):
            pos = np.random.uniform(0, 1, dim)  # random in [-1,1]
            if initial_guess == []:
                pos = self.lower_bounds + pos * (self.upper_bounds - self.lower_bounds)
            else:
                pos = np.random.uniform(-1, 1, dim)  # random in [0,1]
                pos = np.array(initial_guess) + (+pos * (self.upper_bounds) * 0.10)
                pos = np.maximum(pos, self.lower_bounds)
                pos = np.minimum(pos, pos)
            vel = np.random.uniform(
                -self.kappa, self.kappa, dim
            )  # Here we multiply by kappa to allow various starting speeds.
            bee = Bee(pos, vel)
            bee.loss = self.loss_func(bee.position)
            bee.prev_loss = bee.loss + 1.0  # so no pheromone is dropped at t=0
            self.bees.append(bee)
            self.globalIndex += 1

        # Current active pheromones
        self.pheromones = []

        # For logging & potential visualization
        self.history_positions = []  # list of [n_particles x dim] arrays
        self.history_losses = []  # list of [n_particles] arrays
        self.history_pheromones = []  # list of lists-of-pheromones
        self.history_best_loss = []
        self.history_total_weight = []
        self.history_best_solution = []

        self.queen = Bee

    def run(self):
        """
        Runs the deterministic Beehive Optimization for the given number of iterations,
        storing the state at each step for later visualization.
        Also displays a progress bar with iteration, best loss, and best position.
        Constrains bee positions to [lower_bounds, upper_bounds].
        """
        best_bee_global = min(self.bees, key=lambda b: b.loss)
        best_loss_global = best_bee_global.loss
        best_bee_position = best_bee_global.position.copy()
        self.queen = Bee(
            best_bee_global.position.copy(), best_bee_global.velocity.copy(), np.inf
        )

        for t in range(self.n_iterations):
            # 1) identify queen (bee with min loss so far))
            pheromone_positions = np.empty((len(self.pheromones), self.dim))
            pheromone_strengths = np.empty(len(self.pheromones))
            for i in range(len(self.pheromones)):
                pheromone_positions[i] = self.pheromones[i].position
                pheromone_strengths[i] = self.pheromones[i].strength
            temp_queen = min(self.bees, key=lambda b: b.loss)
            wasp = max(self.bees, key=lambda b: b.loss)
            if temp_queen.loss < self.queen.loss:
                self.queen = Bee(
                    temp_queen.position.copy(),
                    temp_queen.velocity.copy(),
                    temp_queen.loss.copy(),
                )

            # 2) update each bee
            k = 0.7
            for bee in self.bees:
                # fluctuating temperature?
                # decreasing temp as func of time
                # increase/decrease fluctations depending on state
                self.c = np.random.normal(
                    0.9 - (k * (t + 1) / self.n_iterations), 0.8 / (t + 1)
                )
                self.c = min(1.2, self.c)
                self.q = np.random.normal(0.1, 0.4 / (t + 1))
                self.q = max(0, self.q)
                old_loss = bee.loss

                # pheromone influence
                pheromone_term = self.compute_pheromone_influence(
                    bee, pheromone_positions, pheromone_strengths
                )

                # queen attraction
                queen_term = (
                    self.queen.position - bee.position
                )  # should queen term be normalized
                # by distanced squared?

                # wasp scattering
                wasp_term = wasp.position - bee.position
                # velocity update: v(t) = rho*v(t-1) + c*pheromone_term + q*queen_term
                bee.velocity = (
                    self.rho * bee.velocity
                    + self.dt * bee.c * pheromone_term
                    + self.dt * bee.q * queen_term
                    - self.dt * self.w * wasp_term
                )

                # position update
                bee.position += bee.velocity * self.dt

                # forcing container 0 to be zero

                # enforce constraints
                bee.position = np.maximum(bee.position, self.lower_bounds)
                bee.position = np.minimum(bee.position, self.upper_bounds)

                # evaluate new loss
                new_loss = self.loss_func(bee.position)
                self.globalIndex += 1
                bee.prev_loss = old_loss
                bee.loss = new_loss

                # if improved, drop pheromone
                # no, now all drop pheremone. If improves, attracts, otherwise, repels.
                #                strength = (bee.prev_loss - bee.loss)
                #                strength = min(np.abs(strength), self.queen.loss)*(strength > 0)
                #                strength = strength*(bee.loss > 0)/bee.loss
                #                self.pheromones.append(Pheromone(bee.position.copy(), strength))

                strength = bee.prev_loss - bee.loss
                strength = strength / np.abs(bee.loss) * 100
                mag_strength = min(np.abs(strength), self.queen.loss)  # *(strength > 0)
                if strength > 0:
                    strength = mag_strength
                else:
                    strength = -mag_strength
                self.pheromones.append(Pheromone(bee.position.copy(), strength))

            # 3) Decay pheromones
            self.decay_pheromones()

            # 4) Store states
            positions = np.array([b.position for b in self.bees])
            losses = np.array([b.loss for b in self.bees])
            self.history_positions.append(positions)
            self.history_losses.append(losses)
            # Make a copy of the pheromone list
            pheromone_snapshot = []
            for p in self.pheromones:
                pheromone_snapshot.append(Pheromone(p.position.copy(), p.strength))
            self.history_pheromones.append(pheromone_snapshot)

            # Update global best if needed
            current_best_idx = np.argmin(losses)
            current_best_loss = losses[current_best_idx]
            if current_best_loss < best_loss_global:
                best_loss_global = current_best_loss
                best_bee_global = self.bees[current_best_idx]
                best_bee_position = self.bees[current_best_idx].position.copy()
            self.history_best_solution.append(best_bee_position)
            self.history_best_loss.append(best_loss_global)

            # 5) Print progress bar / current best
            fraction = (t + 1) / self.n_iterations
            bar_length = 30
            filled_length = int(bar_length * fraction)
            bar = "#" * filled_length + "-" * (bar_length - filled_length)

            # Format the position (truncate if dimension > 5 for neatness)
            if self.dim <= 5:
                best_pos_str = np.round(best_bee_position, 4)
            else:
                # For higher dims, just show first 3 comps + ellipsis
                short_vec = np.round(best_bee_position[:3], 4)
                best_pos_str = f"{short_vec}..."
            msg = (
                f"\rIteration {t + 1}/{self.n_iterations} [{bar}] "
                f" Best Loss: {best_loss_global:.4f} "
                f" Best Pos: {best_pos_str} "
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

        # final newline after completion
        sys.stdout.write("\n")

        return best_bee_position, best_loss_global

    def compute_pheromone_influence(
        self, bee, pheromone_positions, pheromone_strengths
    ):
        """
        Computes the aggregated pheromone influence on the given bee.
        Force from each pheromone p:
           force = p.strength / ||p.position - bee.position||^2
        Then we do a weighted average for direction vectors.
        """
        if len(self.pheromones) == 0:
            return np.zeros(self.dim, dtype=float)

        vectors = pheromone_positions - bee.position
        dist_sq = np.sum(np.abs(vectors) ** 3, axis=1) + 1e-12
        forces = pheromone_strengths / dist_sq
        sum_forces = np.sum(forces)
        if sum_forces < 1e-12:
            return np.zeros(self.dim, dtype=float)
        probs = forces / sum_forces

        vectors = np.array(vectors)
        aggregated = np.sum(vectors.T * probs, axis=1)
        return aggregated

    def decay_pheromones(self):
        """
        Decays all pheromones by factor gamma. Remove those that become too weak.
        """
        new_list = []
        convergence_factor = 1e-6
        if self.queen.loss < convergence_factor * 10:
            min_strength = self.queen.loss / 10
        else:
            min_strength = convergence_factor
        for p in self.pheromones:
            p.strength *= self.gamma
            # perhaps these pheromones should just be removed if they dont have a large force.
            if p.strength > min_strength:  # 1E-5:#min_strength: #self.queen.loss/10:
                new_list.append(p)
        self.pheromones = new_list

    # -------------------------------------------------------------------------
    # 2D Visualization Only
    # -------------------------------------------------------------------------
    def animate_evolution_2d(
        self,
        gif_filename="beehive_evolution.gif",
        xlim=(-5, 5),
        ylim=(-5, 5),
        fps=5,
        resolution=100,
    ):
        """
        If dim=2, creates an animated GIF of all iterations, showing:
          - Heatmap of the loss function
          - Bees (blue dots)
          - Queen (red star)
          - Pheromones (green circles, sized by strength)

        For dim>2, prints a message and returns without visualization.
        """
        if self.dim != 2:
            print("Animation only supported for dim=2. (dim={})".format(self.dim))
            return

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

        scat_phero = ax.scatter([], [], color="green", alpha=0.5, label="Pheromones")
        scat_bees = ax.scatter([], [], color="blue", s=10, label="Bees")
        scat_queen = ax.scatter([], [], color="red", s=120, marker="*", label="Queen")
        scat_wasp = ax.scatter([], [], color="black", s=120, marker="*", label="Wasp")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Beehive Optimization Evolution (2D)")
        ax.legend()

        def init():
            scat_bees.set_offsets(np.empty((0, 2)))
            scat_queen.set_offsets(np.empty((0, 2)))
            scat_phero.set_offsets(np.empty((0, 2)))
            scat_wasp.set_offsets(np.empty((0, 2)))
            return scat_bees, scat_queen, scat_phero, scat_wasp

        def update(frame):
            positions = self.history_positions[frame]
            losses = self.history_losses[frame]
            queen_idx = np.argmin(losses)
            wasp_idx = np.argmax(losses)
            queen_pos = positions[queen_idx]
            wasp_pos = positions[wasp_idx]

            scat_bees.set_offsets(positions)
            scat_queen.set_offsets([queen_pos])
            scat_wasp.set_offsets([wasp_pos])

            # Pheromones
            current_pheromones = self.history_pheromones[frame]
            if len(current_pheromones) > 0:
                p_positions = np.array([ph.position for ph in current_pheromones])
                p_strengths = np.array([ph.strength for ph in current_pheromones])
                sizes = p_strengths * 20
                scat_phero.set_offsets(p_positions)
                scat_phero.set_sizes(sizes)
            else:
                scat_phero.set_offsets(np.empty((0, 2)))

            ax.set_title(f"Iteration {frame + 1}/{self.n_iterations}")
            return scat_bees, scat_queen, scat_phero, scat_wasp

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
        Plots a snapshot of the bees, queen, and pheromones at a given iteration
        overlaid with a heatmap of the loss function. If iteration is None, uses
        the last iteration. This is only for dim=2.
        """
        if self.dim != 2:
            print(
                "visualize_snapshot_2d only works for dim=2. (dim={})".format(self.dim)
            )
            return

        if iteration is None:
            iteration = len(self.history_positions) - 1

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
        queen_idx = np.argmin(losses)
        queen_pos = positions[queen_idx]

        current_pheromones = self.history_pheromones[iteration]
        if len(current_pheromones) > 0:
            p_positions = np.array([ph.position for ph in current_pheromones])
            p_strengths = np.array([ph.strength for ph in current_pheromones])
            sizes = p_strengths * 20
            ax.scatter(
                p_positions[:, 0],
                p_positions[:, 1],
                s=sizes,
                color="green",
                alpha=0.5,
                label="Pheromones",
            )

        ax.scatter(positions[:, 0], positions[:, 1], color="blue", s=10, label="Bees")
        ax.scatter(
            queen_pos[0], queen_pos[1], color="red", s=120, marker="*", label="Queen"
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Snapshot at Iteration {iteration + 1}/{self.n_iterations}")
        ax.legend()
        plt.show()
