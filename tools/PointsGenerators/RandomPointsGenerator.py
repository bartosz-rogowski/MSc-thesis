import math
import numpy as np


def generate_random_points(
        n: int,
        min_value: float = -10,
        max_value: float = 10) -> np.ndarray:
    """Generates 2D points coordinates of range [min_value, max_value)


    :param n: number of points to be generated
    :param min_value: minimal value of range (default: -10)
    :param max_value: maximal value of range (default: 10)
    :return: np.ndarray(shape=(n,2)) of coordinates
    :raise Exception: when min_value equals max_value
    """
    if math.isclose(min_value, max_value):
        raise Exception(f"Values of min_value and max_value cannot be the same")
    return (max_value - min_value) * np.random.random_sample((n, 2)) + min_value
