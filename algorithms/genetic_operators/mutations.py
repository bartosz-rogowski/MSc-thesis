import numpy as np
from numba import njit


@njit
def swap_mutation(solution):
    index1, index2 = np.sort(
        np.random.choice(np.arange(len(solution) - 1), size=2, replace=False)
    )
    solution[index1], solution[index2] = solution[index2], solution[index1]
    if index1 == 0:
        solution[-1] = solution[0]
    return solution


@njit
def displacement_mutation(solution, num_of_elements_to_displace=0):
    if num_of_elements_to_displace < 1:
        num_of_elements_to_displace = np.random.choice(
            np.arange(1, len(solution) // 2),
            size=1,
            replace=True
        )[0]
    start_index, new_start_index = np.sort(
        np.random.choice(
            np.arange(len(solution) - num_of_elements_to_displace),
            size=2,
            replace=False
        )
    )
    #
    subtour: np.ndarray = solution[start_index:start_index + num_of_elements_to_displace].copy()
    #
    for i in range(start_index, new_start_index):
        solution[i] = solution[i + num_of_elements_to_displace]
    new_end_index: int = new_start_index + num_of_elements_to_displace
    solution[new_start_index:new_end_index] = subtour
    solution[-1] = solution[0]
    return solution
