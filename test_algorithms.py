# file: test_algorithms.py

import unittest
import random

from bmssp_algorithm import BMSSPSolver
from baseline_dijkstra import dijkstra_bounded as dijkstra_bounded_multi_source


class TestDijkstraBaseline(unittest.TestCase):
    """Test cases for baseline Dijkstra implementation."""

    def setUp(self):
        self.test_graph = {
            'A': {'B': 1, 'C': 4},
            'B': {'C': 2, 'D': 5},
            'C': {'D': 1},
            'D': {}
        }
    def test_multi_source(self):
        result = dijkstra_bounded_multi_source(self.test_graph, ['A', 'D'], bound=10)
        expected = {'A': 0, 'B': 1, 'C': 3, 'D': 0}
        self.assertEqual(result, expected)

class TestBMSSPAlgorithm(unittest.TestCase):
    """Test cases for BMSSP implementation."""

    def setUp(self):
        self.test_graph = {
            'A': {'B': 1, 'C': 4},
            'B': {'C': 2, 'D': 5},
            'C': {'D': 1},
            'D': {}
        }
        self.solver = BMSSPSolver(self.test_graph)


    def test_multi_source_consistency(self):
        sources = ['A', 'C']
        bound = 8
        bmssp_result = self.solver.solve(sources, bound)
        dijkstra_result = dijkstra_bounded_multi_source(self.test_graph, sources, bound)
        self.assertEqual(set(bmssp_result.keys()), set(dijkstra_result.keys()))
        for vertex in bmssp_result:
            self.assertAlmostEqual(bmssp_result[vertex], dijkstra_result[vertex], places=6)

    def test_bound_enforcement(self):
        result = self.solver.solve(['A'], bound=2)
        expected_vertices = {'A', 'B'}
        self.assertEqual(set(result.keys()), expected_vertices)
        for vertex, distance in result.items():
            self.assertLessEqual(distance, 2)


class TestAlgorithmComparison(unittest.TestCase):
    """Integration tests comparing BMSSP and Dijkstra on various graphs."""

    def test_random_graphs(self):
        test_cases = [
            (50, 3, 15),
            (100, 4, 20),
            (200, 3, 25),
        ]
        for num_nodes, avg_degree, bound in test_cases:
            with self.subTest(nodes=num_nodes, degree=avg_degree, bound=bound):
                graph = {i: {} for i in range(num_nodes)}
                for i in range(num_nodes):
                    for _ in range(avg_degree):
                        neighbor = random.randint(0, num_nodes - 1)
                        if neighbor != i:
                            graph[i][neighbor] = random.randint(1, 10)
                sources = random.sample(range(num_nodes), min(3, num_nodes))
                dijkstra_result = dijkstra_bounded_multi_source(graph, sources, bound)
                bmssp_result = BMSSPSolver(graph).solve(sources, bound)
                self.assertEqual(set(dijkstra_result.keys()), set(bmssp_result.keys()))
                for vertex in dijkstra_result:
                    self.assertAlmostEqual(dijkstra_result[vertex], bmssp_result[vertex], places=5)

    def test_edge_cases(self):
        # Single vertex
        self.assertEqual(BMSSPSolver({'A': {}}).solve(['A'], bound=10), {'A': 0})
        # Empty graph
        self.assertEqual(BMSSPSolver({}).solve([], bound=10), {})
        # Disconnected components
        graph = {'A': {'B': 1}, 'B': {}, 'C': {'D': 1}, 'D': {}}
        self.assertEqual(BMSSPSolver(graph).solve(['A'], bound=10), {'A': 0, 'B': 1})


if __name__ == '__main__':
    random.seed(12345)
    unittest.main()
