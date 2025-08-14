# file: test_algorithms.py

import unittest
import random
from bmssp_algorithm import SpecializedPriorityQueue, BMSSPSolver
from baseline_dijkstra import dijkstra_bounded_multi_source, dijkstra_single_source

class TestSpecializedPriorityQueue(unittest.TestCase):
    """Test cases for the specialized priority queue."""
    
    def setUp(self):
        self.pq = SpecializedPriorityQueue()
    
    def test_bucket_indexing(self):
        """Test the bucket indexing logic."""
        self.assertEqual(self.pq._get_bucket_idx(0), 0)
        self.assertEqual(self.pq._get_bucket_idx(1), 0)
        self.assertEqual(self.pq._get_bucket_idx(2), 1)
        self.assertEqual(self.pq._get_bucket_idx(4), 2)
        self.assertEqual(self.pq._get_bucket_idx(8), 3)
        self.assertEqual(self.pq._get_bucket_idx(15), 3)
        self.assertEqual(self.pq._get_bucket_idx(16), 4)
        print("✅ Bucket indexing test passed")
    
    def test_initialization(self):
        """Test queue initialization."""
        self.pq.initialize(max_bound=64)
        expected_buckets = 7  # log2(64) + 1
        self.assertEqual(len(self.pq.buckets), expected_buckets)
        self.assertTrue(self.pq.is_empty())
        print("✅ Initialization test passed")
    
    def test_batch_operations(self):
        """Test batch decrease-key operations."""
        self.pq.initialize(max_bound=100)
        
        # Add batch updates
        updates = [('A', 5), ('B', 15), ('C', 3), ('D', 25)]
        self.pq.batch_decrease_key(updates)
        
        # Verify lazy processing
        self.assertEqual(len(self.pq.pending_updates), 4)
        
        # Process by checking if empty
        self.assertFalse(self.pq.is_empty())
        self.assertEqual(len(self.pq.pending_updates), 0)
        
        print("✅ Batch operations test passed")
    
    def test_extract_min_order(self):
        """Test that extract_min returns elements in correct order."""
        self.pq.initialize(max_bound=200)
        
        # Add elements in random order
        elements = [('A', 50), ('B', 10), ('C', 30), ('D', 5), ('E', 80)]
        self.pq.batch_decrease_key(elements)
        
        # Extract and verify order
        extracted = []
        while not self.pq.is_empty():
            vertex, dist = self.pq.extract_min()
            extracted.append((vertex, dist))
        
        # Should be sorted by distance
        expected_order = [('D', 5), ('B', 10), ('C', 30), ('A', 50), ('E', 80)]
        self.assertEqual(extracted, expected_order)
        print("✅ Extract min order test passed")
    
    def test_dynamic_updates(self):
        """Test adding updates after some extractions."""
        self.pq.initialize(max_bound=100)
        
        # Initial batch
        self.pq.batch_decrease_key([('A', 20), ('B', 40)])
        
        # Extract one
        vertex1, dist1 = self.pq.extract_min()
        self.assertEqual(vertex1, 'A')
        self.assertEqual(dist1, 20)
        
        # Add more elements
        self.pq.batch_decrease_key([('C', 10), ('D', 50)])
        
        # Extract remaining in order
        vertex2, dist2 = self.pq.extract_min()
        self.assertEqual(vertex2, 'C')
        self.assertEqual(dist2, 10)
        
        print("✅ Dynamic updates test passed")


class TestDijkstraBaseline(unittest.TestCase):
    """Test cases for the baseline Dijkstra implementation."""
    
    def setUp(self):
        self.test_graph = {
            'A': {'B': 1, 'C': 4},
            'B': {'C': 2, 'D': 5},
            'C': {'D': 1},
            'D': {}
        }
    
    def test_single_source(self):
        """Test single-source Dijkstra."""
        result = dijkstra_single_source(self.test_graph, 'A')
        expected = {'A': 0, 'B': 1, 'C': 3, 'D': 4}
        self.assertEqual(result, expected)
        print("✅ Single-source Dijkstra test passed")
    
    def test_bounded_single_source(self):
        """Test bounded single-source Dijkstra."""
        result = dijkstra_single_source(self.test_graph, 'A', bound=3)
        expected = {'A': 0, 'B': 1}  # C and D are beyond bound
        self.assertEqual(result, expected)
        print("✅ Bounded single-source test passed")
    
    def test_multi_source(self):
        """Test multi-source Dijkstra."""
        result = dijkstra_bounded_multi_source(self.test_graph, ['A', 'D'], bound=10)
        expected = {'A': 0, 'B': 1, 'C': 3, 'D': 0}
        self.assertEqual(result, expected)
        print("✅ Multi-source Dijkstra test passed")
    
    def test_unreachable_vertices(self):
        """Test handling of unreachable vertices."""
        disconnected_graph = {
            'A': {'B': 1},
            'B': {},
            'C': {'D': 1},
            'D': {}
        }
        result = dijkstra_single_source(disconnected_graph, 'A')
        expected = {'A': 0, 'B': 1}  # C and D are unreachable
        self.assertEqual(result, expected)
        print("✅ Unreachable vertices test passed")


class TestBMSSPAlgorithm(unittest.TestCase):
    """Test cases for the BMSSP implementation."""
    
    def setUp(self):
        self.test_graph = {
            'A': {'B': 1, 'C': 4},
            'B': {'C': 2, 'D': 5},
            'C': {'D': 1},
            'D': {}
        }
        self.solver = BMSSPSolver(self.test_graph)
    
    def test_basic_functionality(self):
        """Test basic BMSSP functionality."""
        result = self.solver.solve(['A'], bound=10)
        
        # Should find the same vertices as Dijkstra
        dijkstra_result = dijkstra_single_source(self.test_graph, 'A', bound=10)
        
        self.assertEqual(set(result.keys()), set(dijkstra_result.keys()))
        
        # Distances should match
        for vertex in result:
            self.assertAlmostEqual(result[vertex], dijkstra_result[vertex], places=6)
        
        print("✅ Basic BMSSP functionality test passed")
    
    def test_multi_source_consistency(self):
        """Test multi-source BMSSP against Dijkstra."""
        sources = ['A', 'C']
        bound = 8
        
        bmssp_result = self.solver.solve(sources, bound)
        dijkstra_result = dijkstra_bounded_multi_source(self.test_graph, sources, bound)
        
        self.assertEqual(set(bmssp_result.keys()), set(dijkstra_result.keys()))
        
        for vertex in bmssp_result:
            self.assertAlmostEqual(bmssp_result[vertex], dijkstra_result[vertex], places=6)
        
        print("✅ Multi-source consistency test passed")
    
    def test_bound_enforcement(self):
        """Test that the bound is properly enforced."""
        result = self.solver.solve(['A'], bound=2)
        
        # Only A and B should be reachable within bound 2
        expected_vertices = {'A', 'B'}
        self.assertEqual(set(result.keys()), expected_vertices)
        
        # All distances should be within bound
        for vertex, distance in result.items():
            self.assertLess(distance, 2)
        
        print("✅ Bound enforcement test passed")


class TestAlgorithmComparison(unittest.TestCase):
    """Integration tests comparing BMSSP and Dijkstra on various graphs."""
    
    def test_random_graphs(self):
        """Test on multiple random graph structures."""
        test_cases = [
            (50, 3, 15),   # (nodes, avg_degree, bound)
            (100, 4, 20),
            (200, 3, 25),
        ]
        
        for num_nodes, avg_degree, bound in test_cases:
            with self.subTest(nodes=num_nodes, degree=avg_degree, bound=bound):
                # Generate graph
                graph = {}
                for i in range(num_nodes):
                    graph[i] = {}
                    for _ in range(avg_degree):
                        neighbor = random.randint(0, num_nodes - 1)
                        if neighbor != i:
                            graph[i][neighbor] = random.randint(1, 10)
                
                # Random sources
                sources = random.sample(range(num_nodes), min(3, num_nodes))
                
                # Compare results
                dijkstra_result = dijkstra_bounded_multi_source(graph, sources, bound)
                
                bmssp_solver = BMSSPSolver(graph)
                bmssp_result = bmssp_solver.solve(sources, bound)
                
                # Verify consistency
                self.assertEqual(set(dijkstra_result.keys()), set(bmssp_result.keys()),
                               f"Vertex sets differ for {num_nodes}-node graph")
                
                for vertex in dijkstra_result:
                    self.assertAlmostEqual(dijkstra_result[vertex], bmssp_result[vertex], places=5,
                                         msg=f"Distance mismatch for vertex {vertex}")
        
        print("✅ Random graphs comparison test passed")
    
    def test_edge_cases(self):
        """Test edge cases and corner conditions."""
        # Single vertex graph
        single_vertex = {'A': {}}
        solver = BMSSPSolver(single_vertex)
        result = solver.solve(['A'], bound=10)
        self.assertEqual(result, {'A': 0})
        
        # Empty graph
        empty_graph = {}
        solver = BMSSPSolver(empty_graph)
        result = solver.solve([], bound=10)
        self.assertEqual(result, {})
        
        # Disconnected components
        disconnected = {
            'A': {'B': 1},
            'B': {},
            'C': {'D': 1},
            'D': {}
        }
        solver = BMSSPSolver(disconnected)
        result = solver.solve(['A'], bound=10)
        expected = {'A': 0, 'B': 1}
        self.assertEqual(result, expected)
        
        print("✅ Edge cases test passed")


def run_all_tests():
    """Run all test suites with detailed output."""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSpecializedPriorityQueue))
    suite.addTests(loader.loadTestsFromTestCase(TestDijkstraBaseline))
    suite.addTests(loader.loadTestsFromTestCase(TestBMSSPAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestAlgorithmComparison))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Set seed for reproducible tests
    random.seed(12345)
    
    success = run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! Implementation appears correct.")
    else:
        print("\n❌ Some tests failed. Check the implementation.")
        exit(1)