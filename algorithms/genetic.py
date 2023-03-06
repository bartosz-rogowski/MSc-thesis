from typing import Callable
import pygad
from matplotlib import pyplot as plt
import numpy as np
from numba import njit


class GeneticAlgorithm:
    def __init__(self,
                 distance_matrix: np.ndarray,
                 max_iterations: int,
                 **kwargs):
        self.distance_matrix = distance_matrix
        self.MAX_ITERATIONS = max_iterations
        self.n = len(self.distance_matrix)
        gene_space = [i for i in range(self.n)]
        self.generation_fitness_per_epoch = np.zeros(shape=(max_iterations+1,))

        self.ga_instance = pygad.GA(
            fitness_func=self.__calculate_fitness_wrapper(),
            gene_space=gene_space,
            crossover_type=crossover_wrapper,
            mutation_type=swap_mutation,
            num_generations=self.MAX_ITERATIONS,
            # num_genes=self.n, # removable
            gene_type=np.int16,
            on_fitness=self.__on_fitness_wrapper(),
            on_generation=self.__on_generation_wrapper(),
            **kwargs
        )

    def run(self):
        self.ga_instance.run()

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

    def __calculate_fitness_wrapper(self) -> Callable[[np.ndarray, int], float]:
        """Private wrapper method for function calculating fitness.
        It is needed in order to use object variables.

        :return: function calculating fitness fulfilling pygad requirements
        """
        def calculate_fitness(solution: np.ndarray, solution_idx: int) -> float:
            cycle_length: float = self.calculate_cycle_length(solution)
            fitness: float = 1./np.log10(cycle_length)
            return fitness
        return calculate_fitness

    def __on_fitness_wrapper(self):
        def on_fitness(ga_instance, population_fitness):
            if ga_instance.generations_completed == 0:
                fitness = np.average(ga_instance.last_generation_fitness)
                self.generation_fitness_per_epoch[0] = fitness
        return on_fitness

    def __on_generation_wrapper(self):
        def on_generation(ga_instance):
            fitness = np.average(ga_instance.last_generation_fitness)
            epoch = ga_instance.generations_completed
            print("epoch =", epoch)
            self.generation_fitness_per_epoch[epoch] = fitness

        return on_generation

    def get_best_solution(self):
        return self.ga_instance.best_solution()[0]

    def find_shortest_cycle(self):
        self.run()
        return self.get_best_solution()

    def plot_fitness_function(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.ga_instance.best_solutions_fitness, marker='.')
        plt.xlabel("numer pokolenia")
        plt.ylabel("wartość funkcji przystosowania")
        plt.title("Wartość przystosowania w funkcji numeru pokolenia")
        # plt.ylim((-0.05, 1.05))
        plt.grid()
        return plt.gcf()

    def plot_average_fitness_per_epoch(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.generation_fitness_per_epoch, marker='.')
        plt.xlabel("numer pokolenia")
        plt.ylabel("wartość średniego przystosowania populacji")
        plt.title("Średnia wartość przystosowania populacji w funkcji numeru pokolenia")
        # plt.ylim((-0.05, 1.05))
        plt.grid()
        return plt.gcf()


def crossover_wrapper(parents, offspring_size, ga_instance):
    offspring_list = []
    idx = 0
    while idx != offspring_size[0]:
        parent1 = parents[idx % len(parents), :].copy()
        parent2 = parents[(idx + 1) % len(parents), :].copy()

        offspring_list.append(partially_matched_crossover(parent1, parent2))
        # offspring_list.append(partially_matched_crossover(parent2, parent1))
        idx += 1
    return np.array(offspring_list)


def swap_mutation(offspring, ga_instance):
    """Performs swap mutation on the offspring.
    Two indices (genes loci) are drawn (from 0 to len(solution)-1).
    """
    for idx in range(offspring.shape[0]):
        index1, index2 = -1, -1
        while index1 == index2:
            index1 = np.random.randint(low=0, high=offspring.shape[1]-1)
            index2 = np.random.randint(low=0, high=offspring.shape[1]-1)
        temp = offspring[idx, index1]
        offspring[idx, index1] = offspring[idx, index2]
        offspring[idx, index2] = temp
        if index1 == 0 or index2 == 0:
            offspring[idx, -1] = offspring[idx, 0]

    return offspring


@njit
def partially_matched_crossover(parent_1, parent_2, locus1=-1, locus2=-1):
    if locus1 >= locus2:
        locus1, locus2 = np.sort(
            np.random.choice(np.arange(len(parent_1)), size=2, replace=False)
        )

    offspring = np.zeros(
        shape=(len(parent_1),),
        dtype=type(parent_1[0])
    )

    # cycle has the same first and last element - the last one need to be cut
    parent_1 = parent_1[:-1]
    parent_2 = parent_2[:-1]

    offspring[locus1:locus2] = parent_1[locus1:locus2]

    outer_locus_list = np.concatenate((
        np.arange(0, locus1),
        np.arange(locus2, len(parent_1))
    ),)

    for i in outer_locus_list:
        candidate = parent_2[i]
        while candidate in parent_1[locus1:locus2]:
            candidate = parent_2[np.where(parent_1 == candidate)[0][0]]
        offspring[i] = candidate

    # cycle has the same first and last element - the last one have to be the same as first
    offspring[-1] = offspring[0]
    return offspring
