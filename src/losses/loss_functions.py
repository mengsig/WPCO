import numpy as np


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
