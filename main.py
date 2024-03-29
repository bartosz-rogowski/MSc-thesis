import matplotlib.pyplot as plt
import numpy as np
from tools.manage_input_files \
    import generate_coordinates_to_file, load_points_from_file, load_points_from_tsp_file
from tools.generate_starting_cycle import nearest_neighbour
from algorithms.annealing import SimulatedAnnealing
from algorithms.genetic import GeneticAlgorithm
from algorithms.reinforcement import QLearning
from time import perf_counter
from tools.visualisers import Visualiser

if __name__ == '__main__':
    app_start_time = perf_counter()

    precision: int = 3  # of cycle length
    ga_number_of_parents: int = 100  # for genetic algorithm
    # number_of_points: int = 50

    # generate_coordinates_to_file(number_of_points, f"{number_of_points}points")
    points, distance_matrix = load_points_from_tsp_file(
        # f"{number_of_points}points.dat"
        "eil101.tsp"
    )
    if points is None:
        raise Exception("points array is None")
    if distance_matrix is None:
        raise Exception("distances array is None")

    starting_cycle: np.ndarray = nearest_neighbour(distance_matrix, starting_point=0)
    # starting_cycle: np.ndarray = np.arange(number_of_points)
    # np.random.shuffle(starting_cycle)
    # starting_cycle = np.append(starting_cycle, starting_cycle[0])

    # np.savetxt("starting_cycle.dat", starting_cycle, fmt="%i")

    initial_population_array = np.array(
        [nearest_neighbour(distance_matrix) for _ in range(ga_number_of_parents)]
    )
    # initial_population_array = np.zeros(shape=(ga_number_of_parents, number_of_points+1))
    # for i in range(ga_number_of_parents):
    #     starting_cycle: np.ndarray = np.arange(number_of_points)
    #     np.random.shuffle(starting_cycle)
    #     starting_cycle = np.append(starting_cycle, starting_cycle[0])
    #     initial_population_array[i] = starting_cycle
    # algorithm = GeneticAlgorithm(
    #     distance_matrix=distance_matrix,
    #     max_iterations=5000,
    #     initial_population=initial_population_array,
    #     num_parents_mating=ga_number_of_parents//2,
    #     mutation_probability=5e-2,
    #     parent_selection_type="tournament",
    #     keep_elitism=ga_number_of_parents//4,
    #     # parallel_processing=["thread", 4],
    # )

    # algorithm = QLearning(
    #     distance_matrix=distance_matrix,
    #     max_iterations=10_000,
    #     learning_rate=0.1,
    #     discount_rate=0.99,
    # )

    algorithm = SimulatedAnnealing(
        distance_matrix=distance_matrix,
        starting_cycle=starting_cycle,
        max_iterations=distance_matrix.shape[0] ** 2,
        temp_iterations=100,
        start_temperature=1.5,
    )

    alg_start_time = perf_counter()
    shortest_cycle, shortest_cycle_length, cycle_lengths_array, cycle_lengths_iterations_array \
        = algorithm.find_shortest_cycle(precision)
    # shortest_cycle, shortest_cycle_length = algorithm.find_shortest_cycle(precision)
    alg_end_time = perf_counter()
    print(f"{shortest_cycle_length = }")
    # print(f"{shortest_cycle = }")
    print(f"Algorithm ran for {(alg_end_time - alg_start_time):.3f} seconds")
    assert len(set(shortest_cycle)) == len(shortest_cycle) - 1, "More than 1 double element"
    assert shortest_cycle[0] == shortest_cycle[-1], "Not a cycle"

    visualiser = Visualiser(points=points, distance_matrix=distance_matrix)
    visualiser.create_cycle_figure(starting_cycle, title="cykl początkowy")
    visualiser.create_cycle_figure(shortest_cycle, title="znaleziony cykl")

    if cycle_lengths_iterations_array is None:
        plt.figure(figsize=(12, 7))
        plt.title("Długości cykli")
        plt.plot(np.arange(len(cycle_lengths_array)), cycle_lengths_array, '-')
        plt.xlabel("Numer iteracji całkowitej algorytmu")
        plt.ylabel("Długość cyklu")
        plt.grid()
    if False:
        cycle_lengths_array = cycle_lengths_array[cycle_lengths_array > 0]
        cycle_lengths_iterations_array = cycle_lengths_iterations_array[cycle_lengths_iterations_array >= 0]
        assert len(cycle_lengths_array) == len(cycle_lengths_iterations_array)

        plt.figure(figsize=(12, 7))
        plt.title("Długości kolejnych akceptowanych cykli")
        ax1 = plt.subplot(2, 1, 1)
        ax1.set_xlabel("Numer iteracji całkowitej algorytmu")
        ax1.set_ylabel("Długość cyklu")
        ax1.plot(cycle_lengths_iterations_array, cycle_lengths_array, '-')
        ax1.axhline(
            y=cycle_lengths_array[0],
            label=f"początkowa długość = {cycle_lengths_array[0]}",
            color='r',
            linestyle='--'
        )
        ax1.axhline(
            y=cycle_lengths_array[-1],
            label=f"końcowa długość = {cycle_lengths_array[-1]}",
            color='g',
            linestyle='--'
        )
        ax1.axvline(
            x=cycle_lengths_iterations_array[-2],
            label=f"końcowa iteracja = {cycle_lengths_iterations_array[-2]:_}",
            color='gray',
            linestyle='-.'
        )
        ax1.legend()
        ax1.grid()

        fraction: float = 0.05
        # plt.figure(figsize=(12, 8))
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_xlabel("Numer iteracji całkowitej algorytmu")
        ax2.set_ylabel("Długość cyklu")
        end_percent = -1 * int(fraction * len(cycle_lengths_array))
        ax2.plot(
            cycle_lengths_iterations_array[end_percent:-1],
            cycle_lengths_array[end_percent:-1],
            '-'
        )
        # ax2.axhline(
        #     y=cycle_lengths_array[0],
        #     label=f"początkowa długość = {cycle_lengths_array[0]}",
        #     color='r',
        #     linestyle='--'
        # )
        ax2.axhline(
            y=cycle_lengths_array[-1],
            label=f"końcowa długość = {cycle_lengths_array[-1]}",
            color='g',
            linestyle='--'
        )
        ax2.axvline(
            x=cycle_lengths_iterations_array[-2],
            label=f"końcowa iteracja = {cycle_lengths_iterations_array[-2]:_}",
            color='gray',
            linestyle='-.'
        )
        ax2.legend()
        ax2.grid()

    app_end_time = perf_counter()
    print(f"Program ran for {(app_end_time - app_start_time):.3f} seconds")

    # print(cycle_lengths_iterations_array[-100:])
    # print(cycle_lengths_array[-100:])
    # algorithm.plot_fitness_function()
    # algorithm.plot_average_fitness_per_epoch()
    # algorithm.summarize()
    plt.show()
