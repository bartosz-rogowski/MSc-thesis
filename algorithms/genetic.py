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

        self.ga_instance = pygad.GA(
            fitness_func=self.__calculate_fitness_wrapper(),
            gene_space=gene_space,
            crossover_type=crossover_wrapper,
            num_generations=self.MAX_ITERATIONS,
            # num_genes=self.n, # removable
            gene_type=np.int16,
            # allow_duplicate_genes=False,
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
            # print("cycle_length =", cycle_length)
            fitness: float = 1./cycle_length
            return fitness
        return calculate_fitness

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


@njit
def partially_matched_crossover(parent_1, parent_2):

    locus1, locus2 = np.sort(
        np.random.choice(np.arange(len(parent_1)), size=2, replace=False)
    )

    offspring = np.zeros(
        shape=(len(parent_1),),
        dtype=type(parent_1[0])
    )

    offspring[locus1:locus2] = parent_2[locus1:locus2]

    outer_locus_list = np.concatenate((
        np.arange(0, locus1),
        np.arange(locus2, len(parent_1)-1)
    ),)

    for i in outer_locus_list:
        candidate = parent_1[i]
        while candidate in parent_2[locus1:locus2]:
            candidate = parent_1[np.where(parent_2 == candidate)[0][0]]
        offspring[i] = candidate
    print("offspring =", offspring)
    return offspring



# to main.py:
#
# initial_population_array = [nearest_neighbour(distance_matrix) for _ in range(200)]
#     algorithm = GeneticAlgorithm(
#         distance_matrix=distance_matrix,
#         max_iterations=2*distance_matrix.shape[0]**2,
#         num_parents_mating=50,
#         # sol_per_pop=200,
#         mutation_probability=5e-2,
#         parent_selection_type="tournament",
#         mutation_type="swap",
#         keep_elitism=10,
#         initial_population=initial_population_array,
#     )
