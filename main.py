import matplotlib.pyplot as plt
import numpy as np
from tools.manage_input_files \
    import generate_coordinates_to_file, load_points_from_file
from tools.generate_starting_cycle import nearest_neighbour
from algorithms import AlgorithmStrategy
from algorithms.annealing import SimulatedAnnealing, \
    TemperatureCoolingStrategy, SquaredTemperatureCooling
from time import perf_counter
from tools.visualisers import Visualiser


if __name__ == '__main__':
    app_start_time = perf_counter()

    number_of_points: int = 400
    generate_coordinates_to_file(number_of_points, f"{number_of_points}points")
    points, distance_matrix = load_points_from_file(
        f"{number_of_points}points.dat"
    )
    if points is None:
        raise Exception("points array is None")
    if distance_matrix is None:
        raise Exception("distances array is None")

    starting_cycle: np.ndarray = nearest_neighbour(distance_matrix)
    # np.savetxt("starting_cycle.dat", starting_cycle)

    cooling_strategy: TemperatureCoolingStrategy = SquaredTemperatureCooling(
        max_iteration=200
    )
    algorithm: AlgorithmStrategy = SimulatedAnnealing(
        distance_matrix=distance_matrix,
        starting_cycle=starting_cycle,
        max_iterations=90*distance_matrix.shape[0],
        cooling_temperature_strategy=cooling_strategy
    )

    alg_start_time = perf_counter()
    shortest_cycle, shortest_cycle_length = algorithm.find_shortest_cycle()
    alg_end_time = perf_counter()
    print(f"{shortest_cycle_length = :.2f}")
    print(f"Algorithms ran for {(alg_end_time - alg_start_time):.3f} seconds")

    visualiser = Visualiser(points=points, distance_matrix=distance_matrix)
    visualiser.create_cycle_figure(starting_cycle, title="cykl początkowy")
    visualiser.create_cycle_figure(shortest_cycle, title="znaleziony cykl")

    app_end_time = perf_counter()
    print(f"Program ran for {(app_end_time - app_start_time):.3f} seconds")

    plt.show()
