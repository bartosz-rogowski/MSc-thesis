import matplotlib.pyplot as plt
import networkx as nx
from tools.manage_input_files \
    import generate_coordinates_to_file, load_points_from_file
from tools.generate_starting_cycle import nearest_neighbour

if __name__ == '__main__':
    # generate_coordinates_to_file(800, "points3")
    points, distance_matrix = load_points_from_file("points2.dat")
    if points is None:
        raise Exception("points array is None")
    if distance_matrix is None:
        raise Exception("distances array is None")

    # print(distance_matrix)
    # plt.scatter(x=points[:, 0], y=points[:, 1])
    # plt.show()
    print("Starting cycle:")
    starting_cycle = nearest_neighbour(distance_matrix, starting_point=5)
    print(*starting_cycle, sep="\n")
    graph = nx.from_numpy_matrix(distance_matrix)
    nx.draw(graph, pos=points, node_size=60)
    edges = [(starting_cycle[i], starting_cycle[i + 1])
             for i in range(len(starting_cycle) - 1)]
    path = graph.edge_subgraph(edges).copy()
    weights = nx.get_edge_attributes(path, 'weight')
    labels = {}
    for edge, distance in weights.items():
        labels[edge] = round(distance, ndigits=2)
    nx.draw(path, pos=points, node_size=60, edge_color='red', with_labels=True)
    nx.draw_networkx_edge_labels(path, pos=points, edge_labels=labels)
    # nx.draw_networkx_edges(graph, edges, node_size=60, edge_color='r')
    plt.show()
