import numpy as np


def calculate_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Calculates (Euclidean) distance matrix for given points 2D array.

    See
    https://stackoverflow.com/questions/46700326/calculate-distances-between-one-point-in-matrix-from-all-other-points

    :param points: 2D array of coordinates
    :return: np.ndarray of distance matrix
    """
    return np.linalg.norm(points - points[:, None], axis=-1)
