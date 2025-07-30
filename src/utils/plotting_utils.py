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
