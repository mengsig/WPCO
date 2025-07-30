import numpy as np
from numba import njit


# =============================================================================
# Example Loss Function (Rosenbrock generalization)
# =============================================================================
def rosenbrock_nd(x, a=1.0, b=100.0):
    """
    Rosenbrock function generalized to n dimensions.
    """
    total = 0.0
    for i in range(len(x) - 1):
        total += (a - x[i]) ** 2 + b * (x[i + 1] - x[i] ** 2) ** 2
    return total


# =============================================================================
# Weighted loss function for circles
# =============================================================================


@njit()
def compute_included(weighted_matrix, center_x, center_y, radius):
    """
    Compute the weighted sum of values in a matrix covered by a circle.
    Each matrix cell is treated as a continuous area with a uniformly distributed weight.
    The function updates the matrix in-place (sets weights to zero after including them).

    Args:
        weighted_matrix: 2D numpy array of weights.
        center_x, center_y: Center coordinates of the circle (floats).
        radius: Radius of the circle (float).

    Returns:
        included_weight: Sum of weighted values under the circle.
    """
    H, W = weighted_matrix.shape
    start_x = max(int(np.floor(center_x - radius)), 0)
    end_x = min(int(np.ceil(center_x + radius)) + 1, H)
    start_y = max(int(np.floor(center_y - radius)), 0)
    end_y = min(int(np.ceil(center_y + radius)) + 1, W)

    included_weight = 0.0
    r2 = radius * radius

    for i in range(start_x, end_x):
        for j in range(start_y, end_y):
            dx = i + 0.5 - center_x
            dy = j + 0.5 - center_y
            if dx * dx + dy * dy <= r2:
                included_weight += weighted_matrix[i, j]
                if weighted_matrix[i, j] == 0:
                    included_weight += -1
                weighted_matrix[i, j] = 0  # mark as consumed
    return included_weight


def weighted_loss_function(x, radii, weighted_matrix_copy=None):
    """
    Loss = (total_weight / covered_weight)^2
    """
    if weighted_matrix_copy is None:
        weighted_matrix_copy = np.ones((32, 32))
        print(
            "[WARNING]: no weights are being used. Please ensure that you pass a weight_matrix!"
        )

    dim = x.shape[0] // len(radii)
    locations = np.reshape(x, (len(radii), dim))
    weighted_matrix = weighted_matrix_copy.copy()
    weighted_area = 0.0
    for idx in range(len(radii)):
        cx, cy = locations[idx]
        weighted_area += compute_included(weighted_matrix, cx, cy, radii[idx])

    total_weight = np.sum(weighted_matrix_copy)
    if weighted_area == 0:
        return np.inf  # avoid divide-by-zero
    return (total_weight / weighted_area) ** 2
