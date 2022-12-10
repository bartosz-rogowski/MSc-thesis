import numpy as np


def nearest_neighbour(
        distance_matrix: np.ndarray,
        starting_point: int | None = None) -> np.ndarray:
    """Generates starting cycle by finding nearest neighbour of a vertex.
    Starting vertex (point), if not provided, is chosen randomly.

    :param distance_matrix: 2D distance matrix representing a full graph
    :param starting_point: number of element, must be from range
        `(0, len(distance_matrix))`
    :return: np.ndarray of indices in order of visiting,
        is of length `len(distance_matrix)+1`
    """
    number_of_points = distance_matrix.shape[0]
    cycle_length = number_of_points + 1

    if starting_point is None:
        starting_point = np.random.randint(0, number_of_points)
    print(f"{starting_point = }")

    if starting_point < 0 or starting_point > number_of_points:
        starting_point = np.random.randint(0, number_of_points)
        print("Warning: starting point out of range. Choosing it randomly.")

    starting_cycle = -1 * np.ones(shape=(cycle_length,), dtype=int)
    starting_cycle[0] = starting_point

    for i in range(number_of_points-1):
        index_from = starting_cycle[i]
        min_distance = np.infty
        nearest_neighbour_index = -1
        for index_to in range(number_of_points):
            if (distance_matrix[index_from][index_to] - min_distance < 0) \
                    and (index_to not in starting_cycle):
                min_distance = distance_matrix[index_from][index_to]
                nearest_neighbour_index = index_to
        starting_cycle[i + 1] = nearest_neighbour_index

    starting_cycle[-1] = starting_point
    return starting_cycle
