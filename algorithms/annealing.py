from . import AlgorithmStrategy
from . import ABC, abstractmethod
import numpy as np
from typing import Tuple


class TemperatureCoolingStrategy(ABC):
    """Interface for different temperature cooling methods
    """

    @abstractmethod
    def cool_temperature(self) -> None:
        pass

    @abstractmethod
    def get_temperature(self) -> float:
        pass


class SquaredTemperatureCooling(TemperatureCoolingStrategy):
    """Class implementing squared temperature cooling method
    """
    def __init__(self, max_iteration: int, a: float = 1e-3):
        if max_iteration < 0:
            raise ValueError("Iteration number must be positive.")
        self.__iteration = max_iteration
        self.__constant = a
        self.__temperature = self.__constant * np.power(self.__iteration, 2)

    def cool_temperature(self) -> None:
        """Decrements iteration number and cools temperature by the following
        formula: ``self.constant * self.iteration**2``.

        :raise Exception: when ``self.iteration`` becomes negative
        """
        self.__iteration -= 1
        if self.__iteration < 0:
            raise Exception("Exceeded iteration number")
        else:
            self.__temperature = self.__constant * np.power(self.__iteration, 2)

    def get_temperature(self) -> float:
        return self.__temperature


class LinearTemperatureCooling(TemperatureCoolingStrategy):
    """Class implementing linear temperature cooling method
    """
    def __init__(self, max_iteration: int, a: float = 1):
        if max_iteration < 0:
            raise ValueError("Iteration number must be positive.")
        self.__iteration = max_iteration
        self.__constant = a
        self.__temperature = self.__constant * self.__iteration

    def cool_temperature(self) -> None:
        """Decrements iteration number and cools temperature by the following
        formula: ``self.constant * self.iteration``.

        :raise Exception: when ``self.iteration`` becomes negative
        """
        self.__iteration -= 1
        if self.__iteration < 0:
            raise Exception("Exceeded iteration number")
        else:
            self.__temperature = self.__constant * self.__iteration

    def get_temperature(self) -> float:
        return self.__temperature


class SimulatedAnnealing(AlgorithmStrategy):
    """Simulated annealing class
    """
    def __init__(self,
                 distance_matrix: np.ndarray,
                 starting_cycle: np.ndarray,
                 max_iterations: int,
                 cooling_temperature_strategy: TemperatureCoolingStrategy):
        self.distance_matrix = distance_matrix
        self.starting_cycle = starting_cycle
        self.MAX_ITERATIONS = max_iterations
        self.cooling_temperature_strategy = cooling_temperature_strategy

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
            i, j = np.random.randint(len(cycle) - 1, size=2)

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

    def find_shortest_cycle(self) -> Tuple[np.ndarray, float]:
        """Method conducting simmulated annealing

        :return: tuple containing shortest cycle and its length
        """
        temperature: float = self.cooling_temperature_strategy.get_temperature()
        cycle: np.ndarray = self.starting_cycle.copy()
        cycle_length: float = self.calculate_cycle_length(cycle=cycle)
        print(f"starting_cycle_length = {cycle_length:.2f}")
        while temperature > 0:
            for iteration in range(self.MAX_ITERATIONS):
                new_cycle, new_cycle_length = self.create_new_cycle_with_2opt(
                    cycle,
                    cycle_length
                )
                length_diff: float = new_cycle_length - cycle_length

                if length_diff < 0:
                    cycle = new_cycle
                    cycle_length = new_cycle_length
                elif np.random.rand() < np.exp(-length_diff / temperature):
                    cycle = new_cycle
                    cycle_length = new_cycle_length
            # end of for
            self.cooling_temperature_strategy.cool_temperature()
            temperature = self.cooling_temperature_strategy.get_temperature()
        # end of while
        return cycle, cycle_length
