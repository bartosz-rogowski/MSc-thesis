import unittest
import numpy as np
from numpy.testing import assert_array_equal
from algorithms.genetic import GeneticAlgorithm, partially_matched_crossover


class GeneticAlgorithmTest(unittest.TestCase):
    def test_PMX(self):
        parent1 = np.array([8, 4, 7, 3, 6, 2, 5, 1, 9, 0, 8])
        parent2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        result = np.array([0, 7, 4, 3, 6, 2, 5, 1, 8, 9, 0])
        child = partially_matched_crossover(parent1, parent2, locus1=3, locus2=8)
        assert_array_equal(result, child)


if __name__ == '__main__':
    unittest.main()
