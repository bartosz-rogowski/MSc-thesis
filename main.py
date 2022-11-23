import matplotlib.pyplot as plt
import networkx as nx
from tools.manage_input_files \
    import generate_random_points, load_points_from_file

if __name__ == '__main__':
    points, distances = load_points_from_file("middle.dat")
    if points is None:
        raise Exception("points array is None")
    if distances is None:
        raise Exception("distances array is None")

    plt.scatter(x=points[:, 0], y=points[:, 1])
    plt.show()
    graph = nx.from_numpy_matrix(distances)
    nx.draw(graph, pos=points)
    plt.show()
