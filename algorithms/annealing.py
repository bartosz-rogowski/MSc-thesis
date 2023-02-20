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
        ("TEMP_ITERATIONS", numba.int32),
        ("START_TEMPERATURE", numba.float64),
        ("BETA", numba.float64),
        ("END_TEMPERATURE", numba.float64),
        ("temp_iteration", numba.float64),
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
                 temp_iterations: int,
                 start_temperature: int = 10):
        self.distance_matrix = distance_matrix
        self.starting_cycle = starting_cycle
        self.MAX_ITERATIONS = max_iterations
        self.TEMP_ITERATIONS = temp_iterations
        self.START_TEMPERATURE = start_temperature
        self.BETA = 0.85  # constant for exponential cooling schedule
        self.END_TEMPERATURE = self.START_TEMPERATURE * np.power(self.BETA, self.TEMP_ITERATIONS)
        self.temp_iteration = 0
        self.temperature = self.START_TEMPERATURE

    def calculate_cycle_length(self, cycle: np.ndarray):
        """

        :param cycle: np.ndarray of visiting order
        :return: length of a cycle
        """
        length: float = 0.0
        n: int = len(cycle)
        for i in range(n):
            index_from, index_to = cycle[i], cycle[(i + 1) % n]
            length += self.distance_matrix[index_from][index_to]
        return length

    def create_new_cycle_with_2opt(self,
                                   cycle: np.ndarray,
                                   cycle_length: float
                                   ) -> Tuple[np.ndarray, float]:
        """Method that makes 2-opt on a given cycle
        and calculates new cycle length.

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

    def cool_temperature(self, schedule: str = "EXPONENTIAL") -> None:
        """Cools temperature by chosen schedule and increments temperature iteration.
        """
        schedule = schedule.upper()
        self.temp_iteration += 1
        if schedule == "LINEAR":
            a: float = (self.START_TEMPERATURE - self.END_TEMPERATURE)/self.TEMP_ITERATIONS
            b: float = self.END_TEMPERATURE
            self.temperature = a * (self.TEMP_ITERATIONS - self.temp_iteration) + b
        if schedule == "QUADRATIC":
            a: float = -(self.END_TEMPERATURE - self.START_TEMPERATURE) / self.TEMP_ITERATIONS**2
            b: float = 2*(self.END_TEMPERATURE - self.START_TEMPERATURE) / self.TEMP_ITERATIONS
            c: float = self.START_TEMPERATURE
            self.temperature = a * self.temp_iteration**2 + b * self.temp_iteration + c

        if schedule == "EXPONENTIAL":
            self.temperature *= self.BETA

        if schedule == "INVERSE_QUADRATIC":
            a: float = 0.5
            b: float = (self.START_TEMPERATURE/(1+a*self.TEMP_ITERATIONS**2) - self.END_TEMPERATURE) \
                / self.TEMP_ITERATIONS
            self.temperature = self.START_TEMPERATURE/(1. + a*self.temp_iteration**2) - b*self.temp_iteration

    def find_shortest_cycle(self, precision: int = 2) -> Tuple[ndarray, float, ndarray, ndarray]:
        """Method conducting simulated annealing

        :param precision: number of decimals to use while rounding the cycle length
        :return: tuple containing np.ndarray of the shortest cycle, its length
            and np.ndarray of all accepted cycles lengths
        """

        self.temp_iteration = 0
        self.temperature = self.START_TEMPERATURE
        cycle: np.ndarray = self.starting_cycle.copy()
        cycle_length: float = self.calculate_cycle_length(cycle=cycle)
        print("starting_cycle_length =", round(cycle_length, precision))
        cycle_lengths_array: np.ndarray = -1 * np.ones(
            10 * self.TEMP_ITERATIONS * self.distance_matrix.shape[0] ** 2
        )
        cycle_lengths_iterations_array: np.ndarray = -1 * np.ones(
            10 * self.TEMP_ITERATIONS * self.distance_matrix.shape[0] ** 2,
            dtype="int"
        )
        idx: int = 0
        cycle_lengths_array[idx] = round(cycle_length, precision)
        overall_iteration: int = 0
        cycle_lengths_iterations_array[idx] = overall_iteration
        while self.temp_iteration < self.TEMP_ITERATIONS:
            for iteration in range(self.MAX_ITERATIONS):
                new_cycle, new_cycle_length = self.create_new_cycle_with_2opt(
                    cycle,
                    cycle_length
                )
                overall_iteration += 1
                length_diff: float = new_cycle_length - cycle_length

                should_accept_cycle: bool = (length_diff < 0) \
                    or (np.random.rand() < np.exp(-length_diff / self.temperature))
                if should_accept_cycle and length_diff != 0:
                    cycle = new_cycle
                    cycle_length = new_cycle_length
                    idx += 1
                    cycle_lengths_array[idx] = round(cycle_length, precision)
                    cycle_lengths_iterations_array[idx] = overall_iteration
            # end of for
            self.cool_temperature(schedule="EXPONENTIAL")
        # end of while
        idx += 1
        cycle_lengths_array[idx] = round(cycle_length, precision)
        cycle_lengths_iterations_array[idx] = overall_iteration
        print("overall number of accepted cycles =", idx)
        print("overall iterations =", overall_iteration)
        cycle_length = round(cycle_length, precision)
        return cycle, cycle_length, cycle_lengths_array, cycle_lengths_iterations_array
