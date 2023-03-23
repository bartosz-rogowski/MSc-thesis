import unittest
import numpy as np
from numpy.testing import assert_array_equal
from algorithms.genetic import GeneticAlgorithm, partially_matched_crossover, \
    edge_recombination_crossover, order_crossover, untwist_operator


class GeneticAlgorithmTest(unittest.TestCase):
    def test_PMX(self):
        parent1 = np.array([8, 4, 7, 3, 6, 2, 5, 1, 9, 0, 8])
        parent2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        result = np.array([0, 7, 4, 3, 6, 2, 5, 1, 8, 9, 0])
        child = partially_matched_crossover(parent1, parent2, locus1=3, locus2=8)
        assert_array_equal(result, child)

    def test_ER(self):
        parent_1 = np.array([1, 2, 3, 4, 5, 6, 1])
        parent_2 = np.array([2, 4, 3, 1, 5, 6, 2])
        child = edge_recombination_crossover(parent_1, parent_2)
        self.assertEqual(len(set(child)), 6)
        self.assertEqual(child[0], child[-1])

        number_of_points = 500
        parent_1 = np.arange(number_of_points)
        parent_2 = np.arange(number_of_points)
        np.random.shuffle(parent_1)
        np.random.shuffle(parent_2)
        parent_1 = np.append(parent_1, parent_1[0])
        parent_2 = np.append(parent_2, parent_2[0])
        child = edge_recombination_crossover(parent_1, parent_2)
        self.assertEqual(len(set(child)), number_of_points)
        self.assertEqual(child[0], child[-1])

    def test_OX1(self):
        parent1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 1])
        parent2 = np.array([2, 4, 6, 8, 7, 5, 3, 1, 2])
        result = np.array([8, 7, 3, 4, 5, 1, 2, 6, 8])
        child = order_crossover(parent1, parent2, locus1=2, locus2=5)
        assert_array_equal(result, child)

        number_of_points = 500
        parent_1 = np.arange(number_of_points)
        parent_2 = np.arange(number_of_points)
        np.random.shuffle(parent_1)
        np.random.shuffle(parent_2)
        parent_1 = np.append(parent_1, parent_1[0])
        parent_2 = np.append(parent_2, parent_2[0])
        child = order_crossover(parent_1, parent_2)
        self.assertEqual(len(set(child)), number_of_points)
        self.assertEqual(child[0], child[-1])

    def test_untwist_operator(self):
        solution = np.array([2, 3, 4, 9, 8, 7, 6, 5, 1, 2])
        result = np.array([2, 3, 4, 5, 6, 7, 8, 9, 1, 2])
        new_solution = untwist_operator(solution, locus1=3, locus2=7)
        assert_array_equal(result, new_solution)


if __name__ == '__main__':
    unittest.main()
