from .PointsGenerators.RandomPointsGenerator import generate_random_points
import numpy as np
import networkx as nx
import os
from typing import Tuple
from .calculate_distance import calculate_distance_matrix
import tsplib95


def generate_coordinates_to_file(
        n: int,
        filename_without_extension: str) -> None:
    """Saves generated 2D coordinates to input_files directory
    (if it does not exist, it is created) as .dat text file

    :param n: number of points to generate
    :param filename_without_extension: output filename without extension
    """
    points = generate_random_points(n, min_value=-15, max_value=15)
    to_directory = "./input_files/"
    if not os.path.isdir(to_directory):
        os.mkdir(to_directory)
    localisation = to_directory + filename_without_extension + ".dat"
    np.savetxt(localisation, points)


def load_points_from_file(
        filename: str) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Loads points from file and calculates distance matrix based on them

    :param filename:
    :return: np.ndarray of points, np.ndarray of distance matrix
        or tuple of Nones if any exception is raised
    """
    points, distance_matrix = None, None
    to_directory = "./input_files/"
    path = to_directory + filename
    try:
        points = np.loadtxt(path)
        distance_matrix = calculate_distance_matrix(points)
    except Exception as e:
        print(e)
    finally:
        return points, distance_matrix


def load_points_from_tsp_file(
        filename: str) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Loads points from TSPlib file and calculates distance matrix based on them

    :param filename:
    :return: np.ndarray of points, np.ndarray of distance matrix
        or tuple of Nones if any exception is raised
    """
    points, distance_matrix = None, None
    to_directory = "./input_files/"
    path = to_directory + filename
    try:
        tsplib_problem = tsplib95.load(filepath=path)
        graph: nx.Graph = tsplib_problem.get_graph()
        if tsplib_problem.edge_weight_type == 'EUC_2D':
            points = np.array([
                np.array(arr) for arr in list(tsplib_problem.node_coords.values())
            ])
            distance_matrix = calculate_distance_matrix(points)
        else:
            points = np.array(
                list(nx.spring_layout(graph).values())
            )
            distance_matrix = nx.to_numpy_array(graph, dtype="float")
    except Exception as e:
        print(e)
    finally:
        return points, distance_matrix
