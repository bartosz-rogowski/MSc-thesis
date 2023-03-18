import matplotlib.pyplot as plt
import numpy as np
from tools.manage_input_files \
    import generate_coordinates_to_file, load_points_from_file
from tools.generate_starting_cycle import nearest_neighbour
from algorithms.annealing import SimulatedAnnealing
from algorithms.genetic import GeneticAlgorithm
from time import perf_counter
from tools.visualisers import Visualiser

if __name__ == '__main__':
    app_start_time = perf_counter()

    number_of_points: int = 200
    precision: int = 3  # of cycle length

    # generate_coordinates_to_file(number_of_points, f"{number_of_points}points")
    points, distance_matrix = load_points_from_file(
        f"{number_of_points}points.dat"
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

    initial_population_array = [nearest_neighbour(distance_matrix) for _ in range(200)]
    # initial_population_array = np.zeros(shape=(200, number_of_points+1))
    # for i in range(200):
    #     starting_cycle: np.ndarray = np.arange(number_of_points)
    #     np.random.shuffle(starting_cycle)
    #     starting_cycle = np.append(starting_cycle, starting_cycle[0])
    #     initial_population_array[i] = starting_cycle
    algorithm = GeneticAlgorithm(
        distance_matrix=distance_matrix,
        max_iterations=500,
        num_parents_mating=100,
        mutation_probability=5e-2,
        parent_selection_type="tournament",
        keep_elitism=50,
        initial_population=initial_population_array,
        # parallel_processing=["thread", 4],
    )

    # algorithm = SimulatedAnnealing(
    #     distance_matrix=distance_matrix,
    #     starting_cycle=starting_cycle,
    #     max_iterations=distance_matrix.shape[0] ** 2,
    #     temp_iterations=100,
    #     start_temperature=0.1,
    # )

    alg_start_time = perf_counter()
    # shortest_cycle, shortest_cycle_length, cycle_lengths_array, cycle_lengths_iterations_array \
    #     = algorithm.find_shortest_cycle(precision)
    shortest_cycle, shortest_cycle_length = algorithm.find_shortest_cycle(precision)
    alg_end_time = perf_counter()
    print(f"{shortest_cycle_length = }")
    # print(f"{shortest_cycle = }")
    print(f"Algorithm ran for {(alg_end_time - alg_start_time):.3f} seconds")
    assert len(set(shortest_cycle)) == len(shortest_cycle) - 1, "More than 1 double element"
    assert shortest_cycle[0] == shortest_cycle[-1], "Not a cycle"

    visualiser = Visualiser(points=points, distance_matrix=distance_matrix)
    # visualiser.create_cycle_figure(starting_cycle, title="cykl początkowy")
    visualiser.create_cycle_figure(shortest_cycle, title="znaleziony cykl")

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

        fraction: float = 0.001
        # plt.figure(figsize=(12, 8))
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_xlabel("Numer iteracji całkowitej algorytmu")
        ax2.set_ylabel("Długość cyklu")
        end_percent = -1 * int(fraction * len(cycle_lengths_array))
        ax2.plot(
            cycle_lengths_iterations_array[end_percent:],
            cycle_lengths_array[end_percent:],
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
    algorithm.plot_fitness_function()
    algorithm.plot_average_fitness_per_epoch()
    # algorithm.summarize()
    plt.show()
