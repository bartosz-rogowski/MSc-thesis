from typing import Callable
import pygad
from matplotlib import pyplot as plt
import numpy as np
from numba import njit
from .genetic_operators import crossovers, mutations

class GeneticAlgorithm:
    def __init__(self,
                 distance_matrix: np.ndarray,
                 max_iterations: int,
                 initial_population: np.ndarray,
                 **kwargs):
        self.distance_matrix = distance_matrix
        self.MAX_ITERATIONS = max_iterations
        self.n = len(self.distance_matrix)
        gene_space = [i for i in range(self.n)]
        self.generation_fitness_per_epoch = np.zeros(shape=(max_iterations + 1,))
        self.best_solution = np.empty_like(initial_population[0])
        self.best_solution_fitness: float = 0.0
        self.best_solution_generation: int = -1

        self.ga_instance = pygad.GA(
            initial_population=initial_population,
            fitness_func=self.__calculate_fitness_wrapper(),
            gene_space=gene_space,
            num_generations=self.MAX_ITERATIONS,
            gene_type=np.int16,
            crossover_type=self.__crossover_wrapper(
                crossover_func=crossovers.partially_matched_crossover,
            ),
            mutation_type=self.__mutation_wrapper(
                mutation_func=mutations.swap_mutation,
                # num_of_elements_to_displace=1  # displacement_mutation arg
            ),
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
            fitness: float = 1. / cycle_length
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
            # if epoch % 100 == 0:
            #     print("epoch =", epoch)
            self.generation_fitness_per_epoch[epoch] = fitness

            # applying untwist operator
            # NOTE: uncommenting this block is easier than providing additional variable
            idx: int = np.random.choice(len(ga_instance.population), size=1, replace=True)[0]
            solution: np.ndarray = ga_instance.population[idx]
            tries: int = 10
            while tries > 0:
                locus1, locus2 = np.random.choice(np.arange(1, len(solution) - 1), size=2, replace=False)
                condition: bool = self.distance_matrix[solution[locus1], solution[locus1 - 1]] \
                                  + self.distance_matrix[solution[locus2 + 1], solution[locus2]] \
                                  > self.distance_matrix[solution[locus2], solution[locus1 - 1]] \
                                  + self.distance_matrix[solution[locus2 + 1], solution[locus1]]
                if condition:
                    # print("applying untwisting")
                    ga_instance.population[idx] = untwist_operator(solution, locus1=locus1, locus2=locus2)
                    break
                tries -= 1
            # end of while

            # saving best solution
            solution, fitness = self.ga_instance.best_solution()[:2]
            if fitness > self.best_solution_fitness:
                self.best_solution = solution
                self.best_solution_fitness = fitness
                self.best_solution_generation = epoch

        return on_generation

    def __crossover_wrapper(self, crossover_func):
        def crossover(parents, offspring_size, ga_instance):
            offspring_list = []
            idx = 0
            while idx != offspring_size[0]:
                parent1 = parents[idx % len(parents), :].copy()
                parent2 = parents[(idx + 1) % len(parents), :].copy()
                offspring1 = crossover_func(parent1, parent2)
                # offspring2 = crossover_func(parent2, parent1)

                # if self.calculate_cycle_length(offspring1) < self.calculate_cycle_length(offspring2):
                offspring_list.append(offspring1)
                # else:
                #     offspring_list.append(offspring2)
                idx += 1
            return np.array(offspring_list)

        return crossover

    def __mutation_wrapper(self, mutation_func, **kwargs):
        def mutation(offspring, ga_instance):
            for idx in range(offspring.shape[0]):
                offspring[idx] = mutation_func(offspring[idx], **kwargs)
            return offspring

        return mutation

    def get_best_solution(self):
        # return self.ga_instance.best_solution()[0]
        return self.best_solution, self.best_solution_fitness, self.best_solution_generation

    def find_shortest_cycle(self, precision: int):
        self.run()
        # best_solution = self.get_best_solution()
        best_solution: np.ndarray = self.get_best_solution()[0]
        length: float = round(self.calculate_cycle_length(best_solution), precision)
        return best_solution, length

    def plot_fitness_function(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.ga_instance.best_solutions_fitness, marker='.')
        plt.xlabel("numer pokolenia")
        plt.ylabel("wartość funkcji przystosowania")
        plt.title("Wartość najlepszego przystosowania w funkcji numeru pokolenia")
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

    def summarize(self, **kwargs):
        self.ga_instance.summary(**kwargs)


@njit
def untwist_operator(solution, locus1=-1, locus2=-1):
    solution_length = len(solution[:-1])
    if locus1 >= locus2:
        locus1, locus2 = np.sort(
            np.random.choice(np.arange(solution_length), size=2, replace=False)
        )
    # new_solution = solution.copy()
    # for _idx in range(locus2 - locus1 + 1):
    #     idx = locus2 + locus1 - locus1 - _idx
    #     new_solution[locus1 + _idx] = solution[idx]

    for _idx in range((locus2 - locus1 + 1) // 2):
        # idx = locus2 + locus1 - locus1 - _idx
        value = solution[locus1 + _idx]
        solution[locus1 + _idx] = solution[locus2 - _idx]
        solution[locus2 - _idx] = value

    solution[-1] = solution[0]
    return solution
