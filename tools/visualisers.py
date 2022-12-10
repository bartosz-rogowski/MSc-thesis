import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple


class Visualiser:
    """Helper class for visualising graph and cycles on it.
    """
    def __init__(self, points, distance_matrix):
        self.points = points
        self.distance_matrix = distance_matrix
        self.graph = nx.from_numpy_matrix(self.distance_matrix)

    def create_graph_figure(self,
                            figsize: Tuple[float, float] = (8, 6),
                            title: str = "",
                            node_size: int = 1,
                            ) -> plt.Figure:
        """Creates a handle to a figure containing plot of a graph

        :param figsize: size of a figure
        :param title: name attached to the figure
        :param node_size: size of every node plotted
        :return: handle to the figure
        """
        fig: plt.Figure = plt.figure(figsize=figsize)
        plt.title(title)
        nx.draw(self.graph, pos=self.points, node_size=node_size)
        return fig

    def create_cycle_figure(self, cycle: np.ndarray, *,
                            draw_on_graph: bool = False,
                            figsize: Tuple[float, float] = (8, 6),
                            title: str = "",
                            node_size: int = 1,
                            edge_color: str = "red",
                            annotate: bool = False,
                            ) -> plt.Figure:
        """Creates a handle to a figure containing plot of a cycle on the graph

        :param cycle: np.ndarray of visiting order
        :param draw_on_graph: if True, cycle is drawn on the top of the graph
        :param figsize: size of a figure
        :param title: name attached to the figure
        :param node_size: size of every node plotted
        :param edge_color: color of every node plotted
        :param annotate: if True, labels (containing distances)
            are added to the plot
        :return: handle to the figure
        """
        fig: plt.Figure = plt.figure(figsize=figsize)
        plt.title(title)
        if draw_on_graph:
            nx.draw(self.graph, pos=self.points, node_size=node_size)
        edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
        path = self.graph.edge_subgraph(edges).copy()
        nx.draw(
            path,
            pos=self.points,
            node_size=node_size,
            edge_color=edge_color,
            with_labels=annotate
        )
        if annotate:
            weights = nx.get_edge_attributes(path, 'weight')
            labels = {}
            for edge, distance in weights.items():
                labels[edge] = round(distance, ndigits=2)
            nx.draw_networkx_edge_labels(
                path,
                pos=self.points,
                edge_labels=labels
            )
        return fig
