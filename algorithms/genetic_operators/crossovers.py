import numpy as np
from numba import njit


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
    ), )
    mapping: dict = {}
    for locus in range(locus1, locus2):
        mapping[parent_1[locus]] = parent_2[locus]
    for i in outer_locus_list:
        candidate = parent_2[i]
        while candidate in parent_1[locus1:locus2]:
            candidate = mapping[candidate]
        offspring[i] = candidate

    # cycle has the same first and last element - the last one have to be the same as first
    offspring[-1] = offspring[0]
    return offspring


@njit
def edge_recombination_crossover(parent_1, parent_2):
    # cycle has the same first and last element - the last one need to be cut
    parent_1 = parent_1[:-1]
    parent_2 = parent_2[:-1]

    parents_length: int = len(parent_1)

    child = -1 * np.ones(
        shape=(parents_length + 1,),
        dtype=type(parent_1[0])
    )

    direct_neighbours_dict: dict = {}
    for i, edge in enumerate(parent_1):
        j = np.where(parent_2 == edge)[0][0]
        neighbours = np.array([
            parent_1[(i - 1) % parents_length],
            parent_1[(i + 1) % parents_length],
            parent_2[(j - 1) % parents_length],
            parent_2[(j + 1) % parents_length],
        ], dtype="int")
        direct_neighbours_dict[edge] = np.unique(neighbours)

    i: int = 0
    edge = np.random.choice(
        np.array([parent_1[0], parent_2[0]]),
        size=1,
        replace=True
    )[0]
    while i < parents_length:
        child[i] = edge
        direct_neighbours_list = direct_neighbours_dict.pop(edge)

        for key, value in direct_neighbours_dict.items():
            idx: np.ndarray = np.where(value == edge)[0]  # indices array
            if idx.size > 0:
                direct_neighbours_dict[key] = np.delete(
                    direct_neighbours_dict[key],
                    idx[0]  # it is an array
                )

        if len(direct_neighbours_list) > 0:
            neighbour_edge_neighbours_count = np.zeros(
                shape=(len(direct_neighbours_list), 2),
                dtype="int"
            )
            for idx, neighbour_edge in enumerate(direct_neighbours_list):
                neighbour_edge_neighbours_count[idx] = np.array(
                    [neighbour_edge, len(direct_neighbours_dict[neighbour_edge])]
                )
            fewest_neighbours = neighbour_edge_neighbours_count[:, 1].min()
            candidates = []
            for neighbour_edge, neighbours_number in neighbour_edge_neighbours_count:
                if neighbours_number == fewest_neighbours:
                    candidates.append(neighbour_edge)
            edge = np.random.choice(np.array(candidates), size=1, replace=True)[0]
        else:
            candidates = []  # parent_1[~np.isin(parent_1, child)]
            for edge in parent_1:
                if edge not in child:
                    candidates.append(edge)
            if len(candidates) == 0:
                break
            edge = np.random.choice(np.array(candidates), size=1, replace=True)[0]
        i += 1

    # cycle has the same first and last element - the last one have to be the same as first
    child[-1] = child[0]
    return child


@njit
def order_crossover(parent_1, parent_2, locus1=-1, locus2=-1):
    parents_length: int = len(parent_1)

    if locus1 >= locus2:
        locus1, locus2 = np.sort(
            np.random.choice(np.arange(parents_length), size=2, replace=False)
        )

    child = -1 * np.ones(
        shape=(parents_length,),
        dtype=type(parent_1[0])
    )

    # cycle has the same first and last element - the last one need to be cut
    parent_1 = parent_1[:-1]
    parent_2 = parent_2[:-1]

    child[locus1:locus2] = parent_1[locus1:locus2]

    idx_in_parent_2: int = locus2

    for i in range(parents_length - 1 - (locus2 - locus1)):
        idx: int = (locus2 + i) % (parents_length - 1)
        value: int = parent_2[idx_in_parent_2]
        while value in child:
            idx_in_parent_2 = (idx_in_parent_2 + 1) % (parents_length - 1)
            value = parent_2[idx_in_parent_2]
        child[idx] = value

    child[-1] = child[0]
    return child
