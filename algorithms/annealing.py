import numba
import numpy as np
from typing import Tuple
from numba.experimental import jitclass
from numpy import ndarray


@jitclass(
    spec=[
        ("distance_matrix", numba.float64[:, :]),
        ("starting_cycle", numba.int32[:]),
        ("MAX_ITERATIONS", numba.int32),
        ("temp_iteration", numba.int32),
        ("constant", numba.float64),
        ("temperature", numba.float64),
    ]
)
class SimulatedAnnealing:
    """Simulated annealing class
    """

    def __init__(self,
                 distance_matrix: np.ndarray,
                 starting_cycle: np.ndarray,
                 max_iterations: int,
                 temp_iteration: int):
        self.distance_matrix = distance_matrix
        self.starting_cycle = starting_cycle
        self.MAX_ITERATIONS = max_iterations
        self.temp_iteration = temp_iteration
        self.constant = 1e-3
        self.temperature = self.constant * np.power(self.temp_iteration, 2)

    def calculate_cycle_length(self, cycle: np.ndarray):
        """

        :param cycle: np.ndarray of visiting order
        :return: length of a cycle
        """
        length: float = 0.0
        for i in range(len(cycle) - 1):
            index_from, index_to = cycle[i], cycle[i + 1]
            length += self.distance_matrix[index_from][index_to]
        return length

    def create_new_cycle_with_2opt(self,
                                   cycle: np.ndarray,
                                   cycle_length: float
                                   ) -> Tuple[np.ndarray, float]:
        """Method that makes 2-opt on a given cycle
        and calcucalates new cycle length.

        :param cycle: np.ndarray of visiting order
        :param cycle_length:
        :return: tuple containing new cycle and its length
        """
        new_cycle = cycle.copy()
        new_cycle_length = cycle_length
        i, j = -1, -1
        while abs(i - j) <= 1 or abs(i - j) == len(cycle) - 1:
            i = np.random.randint(len(cycle) - 1)
            j = np.random.randint(len(cycle) - 1)

        if i > j:
            i, j = j, i

        a, b = i, i + 1
        c, d = j, j + 1
        new_cycle_length -= self.distance_matrix[cycle[a]][cycle[b]]
        new_cycle_length -= self.distance_matrix[cycle[c]][cycle[d]]
        new_cycle_length += self.distance_matrix[cycle[a]][cycle[c]]
        new_cycle_length += self.distance_matrix[cycle[b]][cycle[d]]
        new_cycle[b:d] = new_cycle[b:d][::-1]
        return new_cycle, new_cycle_length

    def cool_temperature(self) -> None:
        """Decrements iteration number and cools temperature by the following
        formula: ``self.constant * self.iteration**2``.
        """
        self.temp_iteration -= 1
        self.temperature = self.constant * np.power(self.temp_iteration, 2)

    def find_shortest_cycle(self) -> tuple[ndarray, float, ndarray, ndarray]:
        """Method conducting simulated annealing

        :return: tuple containing np.ndarray of the shortest cycle, its length
            and np.ndarray of all accepted cycles lengths
        """

        cycle: np.ndarray = self.starting_cycle.copy()
        cycle_length: float = self.calculate_cycle_length(cycle=cycle)
        print("starting_cycle_length =", round(cycle_length, 2))
        cycle_lengths_array: np.ndarray = -1*np.ones(
            10*self.temp_iteration*self.distance_matrix.shape[0]**2
        )
        cycle_lengths_iterations_array: np.ndarray = -1 * np.ones(
            10 * self.temp_iteration * self.distance_matrix.shape[0] ** 2,
            dtype="int"
        )
        idx: int = 0
        cycle_lengths_array[idx] = round(cycle_length, 2)
        overall_iteration: int = 0
        cycle_lengths_iterations_array[idx] = overall_iteration
        while self.temperature > 0:
            for iteration in range(self.MAX_ITERATIONS):
                new_cycle, new_cycle_length = self.create_new_cycle_with_2opt(
                    cycle,
                    cycle_length
                )
                overall_iteration += 1
                length_diff: float = new_cycle_length - cycle_length

                should_accept_cycle: bool = length_diff < 0 \
                    or np.random.rand() < np.exp(-length_diff/self.temperature)
                if should_accept_cycle:
                    cycle = new_cycle
                    cycle_length = new_cycle_length
                    idx += 1
                    cycle_lengths_array[idx] = round(cycle_length, 2)
                    cycle_lengths_iterations_array[idx] = overall_iteration
            # end of for
            self.cool_temperature()
        # end of while
        cycle_lengths_array[idx] = round(cycle_length, 2)
        cycle_lengths_iterations_array[idx] = overall_iteration
        print("overall number of accepted cycles =", idx)
        print("overall iterations =", overall_iteration)
        return cycle, cycle_length, cycle_lengths_array, cycle_lengths_iterations_array
