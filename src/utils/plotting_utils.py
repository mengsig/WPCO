import matplotlib.pyplot as plt


class plot_defaults:
    @staticmethod
    def defaultPlotting():
        plt.rcParams.update(
            {
                "font.size": 14,
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "figure.figsize": (8, 6),
                "axes.grid": True,
                "grid.alpha": 0.5,
                "lines.linewidth": 2,
                "lines.markersize": 6,
                "savefig.dpi": 1200,
                "savefig.format": "png",
            }
        )


def plot_losses(history, history_best=None, savename=None, showPlot=False):
    plot_defaults()
    import numpy as np

    fig, ax = plt.subplots()
    hist = np.array(history)
    for i in range(hist.shape[1]):
        ax.semilogy(hist[:, i], ":")
    if history_best is not None:
        ax.semilogy(history_best, color="black")
    ax.set_xlabel("iteration no.")
    ax.set_ylabel("loss")
    plt.tight_layout()
    if savename is not None:
        fig.savefig(savename, dpi=300)
    if showPlot:
        plt.show()


def plot_heatmap_with_circles(weighted_matrix, pos, radii, savename=None):
    """
    Plots a heatmap with transparent circles on top.

    Args:
        weighted_matrix (2D array): The heatmap matrix.
        x, y (list of floats): Coordinates of the circle centers.
        radii (list of floats): Radii of the circles.
        savename (str): Path to save the image.
    """
    import numpy as np
    from matplotlib.patches import Circle

    plot_defaults()

    pos_reshape = np.reshape(pos, (len(radii), 2))
    x = pos_reshape[:, 0]
    y = pos_reshape[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the heatmap
    heatmap = ax.imshow(
        weighted_matrix, cmap="hot", interpolation="nearest", origin="upper"
    )

    # Add transparent circles
    for xi, yi, ri in zip(x, y, radii):
        circle = Circle((xi, yi), radius=ri, color="blue", alpha=0.5)
        ax.add_patch(circle)

    # Optionally add colorbar
    plt.colorbar(heatmap, ax=ax)

    ax.set_xlim(0, weighted_matrix.shape[1])
    ax.set_ylim(weighted_matrix.shape[0], 0)  # Flip y-axis to match imshow
    ax.set_aspect("equal")

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=300)
    plt.close()
