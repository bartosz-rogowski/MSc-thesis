import numba
import numpy as np
from numba import njit
from numba.experimental import jitclass


@jitclass(
    spec=[
        ("distance_matrix", numba.float64[:, :]),
        ("learning_rate", numba.float64),
        ("discount_rate", numba.float64),
        ("MAX_ITERATIONS", numba.int32),
    ]
)
class QLearning:
    def __init__(self,
                 distance_matrix: np.ndarray,
                 learning_rate: float,
                 discount_rate: float,
                 max_iterations: int
                 ):
        self.distance_matrix = distance_matrix
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.MAX_ITERATIONS = max_iterations

    def calculate_cycle_length(self, cycle: np.ndarray) -> float:
        """

        :param cycle: np.ndarray of visiting order
        :return: length of a cycle
        """
        length: float = 0.0
        for i in range(len(cycle) - 1):
            index_from, index_to = cycle[i], cycle[i + 1]
            length += self.distance_matrix[index_from][index_to]
        return length

    def find_shortest_cycle(self, precision: int):
        n: int = len(self.distance_matrix)
        iteration: int = 0
        epsilon_start: float = 0.85
        epsilon_end: float = 0.99
        delta: float = 1.
        theta_1: float = 1.
        theta_2: float = 0.9
        cycle_length: float = np.infty
        best_cycle_length: float = cycle_length
        cycle: np.ndarray = -1*np.ones(shape=(n+1), dtype="int")
        best_cycle: np.ndarray = cycle.copy()
        cycle_lengths_array: np.ndarray = np.zeros(shape=(self.MAX_ITERATIONS,))

        q_table: np.ndarray = np.zeros(shape=(n, n))
        reward_matrix: np.ndarray = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    reward_matrix[i, j] = -self.distance_matrix[i, j]
                    q_table[i, j] = reward_matrix[i, j]
                else:
                    reward_matrix[i, j] = -np.infty

        while iteration < self.MAX_ITERATIONS:
            epsilon: float = epsilon_start + \
                (epsilon_end - epsilon_start) * np.sqrt(iteration/self.MAX_ITERATIONS)
            # if iteration % 10_000 == 0:
            #     print("iteration =", iteration)
            cycle[0] = np.random.choice(np.arange(n), size=1, replace=True)[0]
            possible_cities = setdiff1d_nb(
                np.arange(n).astype(np.int32),
                np.array([cycle[0]]).astype(np.int32)
            )
            current_city: int = cycle[0]
            next_city: int = -1
            t: int = 1
            while t < n:
                arg_sorted = np.argsort(q_table[current_city][possible_cities])
                if np.random.rand() < epsilon:
                    # idx = q_table[current_city][possible_cities].argmax()
                    idx = arg_sorted[-1]
                    next_city = possible_cities[idx]
                else:
                    if np.random.rand() < epsilon and t < n-1:
                        nth_best_idx = 2 #np.random.choice(
                        #     np.arange(2, min(5, n-t)+1),
                        #     size=1,
                        #     replace=True
                        # )[0]
                        idx = arg_sorted[-nth_best_idx]
                        next_city = possible_cities[idx]
                    else:
                        next_city = np.random.choice(
                            possible_cities,
                            size=1,
                            replace=True
                        )[0]
                # d_max: float = self.distance_matrix[next_city].max()
                # taking second min element because the first is 0 (diagonal elem.)
                d_min: float = np.sort(self.distance_matrix[next_city])[1]
                d_avg: float = np.average(self.distance_matrix[next_city])
                additional_reward = (1. - theta_1)*theta_2/d_min * (1. - theta_1)*(1. - theta_2)/d_avg
                # additional_reward = 10*theta_2/d_min * (1. - theta_2)/d_avg
                reward_matrix[current_city, next_city] = theta_1 * reward_matrix[current_city, next_city] \
                    + additional_reward
                # print(additional_reward)
                if t < n-1:
                    max_q_next_city: float = q_table[next_city][possible_cities].max()
                    q_table[current_city, next_city] = (1 - self.learning_rate) * q_table[current_city, next_city] \
                        + self.learning_rate * (reward_matrix[current_city, next_city] + self.discount_rate * max_q_next_city)
                else:
                    q_table[current_city, next_city] = (1 - self.learning_rate) * q_table[current_city, next_city] \
                                                       + self.learning_rate * reward_matrix[current_city, next_city]
                possible_cities = setdiff1d_nb(
                    possible_cities,
                    np.array([next_city]).astype(np.int32)
                )
                cycle[t] = next_city
                current_city = next_city
                t += 1
            # end of while

            cycle[-1] = cycle[0]
            cycle_length = round(self.calculate_cycle_length(cycle), precision)
            cycle_lengths_array[iteration] = cycle_length
            # print(f"{iteration=}: {cycle_length = :.3f} - {cycle = }")
            if cycle_length < best_cycle_length:
                best_cycle = cycle.copy()
                best_cycle_length = cycle_length
            iteration += 1
            # epsilon *= 0.999
        # end of while (iterations)

        return best_cycle, best_cycle_length, cycle_lengths_array, None


@njit('int32[:](int32[:], int32[:])')
def setdiff1d_nb(arr1, arr2):
    delta = set(arr2)

    # : build the result
    result = np.empty(len(arr1), dtype=arr1.dtype)
    j = 0
    for i in range(arr1.shape[0]):
        if arr1[i] not in delta:
            result[j] = arr1[i]
            j += 1
    return result[:j]