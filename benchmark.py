import random
import time
import statistics
from collections import defaultdict

# Assuming the corrected implementations are in separate files
from bmssp_algorithm import BMSSPSolver
from baseline_dijkstra import dijkstra_bounded_multi_source

class GraphGenerator:
    """Utility class for generating test graphs with different characteristics."""
    
    @staticmethod
    def generate_erdos_renyi(num_nodes, edge_probability=0.1, max_weight=10):
        """Generate an Erdős–Rényi random graph."""
        graph = defaultdict(dict)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < edge_probability:
                    weight = random.randint(1, max_weight)
                    graph[i][j] = weight
        
        return dict(graph)
    
    @staticmethod
    def generate_sparse_connected(num_nodes, avg_degree=4, max_weight=10):
        """Generate a sparse but connected graph."""
        graph = defaultdict(dict)
        
        # First, create a spanning tree to ensure connectivity
        for i in range(1, num_nodes):
            parent = random.randint(0, i - 1)
            weight = random.randint(1, max_weight)
            graph[parent][i] = weight
            # Make it bidirectional for better connectivity
            graph[i][parent] = weight
        
        # Add additional random edges to reach target degree
        edges_needed = (num_nodes * avg_degree) // 2 - (num_nodes - 1)
        
        for _ in range(edges_needed):
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v and v not in graph[u]:
                weight = random.randint(1, max_weight)
                graph[u][v] = weight
        
        return dict(graph)
    
    @staticmethod
    def generate_grid_graph(width, height, max_weight=5):
        """Generate a 2D grid graph (good for testing hierarchical algorithms)."""
        graph = defaultdict(dict)
        
        def node_id(x, y):
            return f"{x},{y}"
        
        for x in range(width):
            for y in range(height):
                current = node_id(x, y)
                
                # Add edges to adjacent cells
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor = node_id(nx, ny)
                        weight = random.randint(1, max_weight)
                        graph[current][neighbor] = weight
        
        return dict(graph)


class BenchmarkRunner:
    """Runs comprehensive benchmarks comparing BMSSP and Dijkstra."""
    
    def __init__(self):
        self.results = []
    
    def verify_correctness(self, graph, sources, bound):
        """Verify that both algorithms produce the same results."""
        print("Verifying correctness...")
        
        # Run Dijkstra
        dijkstra_result = dijkstra_bounded_multi_source(graph, sources, bound)
        
        # Run BMSSP
        bmssp_solver = BMSSPSolver(graph)
        bmssp_result = bmssp_solver.solve(sources, bound)
        
        # Compare results
        dijkstra_vertices = set(dijkstra_result.keys())
        bmssp_vertices = set(bmssp_result.keys())
        
        if dijkstra_vertices != bmssp_vertices:
            print(f"❌ Vertex sets differ!")
            print(f"   Dijkstra found: {len(dijkstra_vertices)} vertices")
            print(f"   BMSSP found: {len(bmssp_vertices)} vertices")
            print(f"   Only in Dijkstra: {dijkstra_vertices - bmssp_vertices}")
            print(f"   Only in BMSSP: {bmssp_vertices - dijkstra_vertices}")
            return False
        
        # Check distance consistency
        for vertex in dijkstra_vertices:
            if abs(dijkstra_result[vertex] - bmssp_result[vertex]) > 1e-6:
                print(f"❌ Distance mismatch for {vertex}:")
                print(f"   Dijkstra: {dijkstra_result[vertex]}")
                print(f"   BMSSP: {bmssp_result[vertex]}")
                return False
        
        print(f"✅ Correctness verified! Both algorithms found {len(dijkstra_vertices)} vertices.")
        return True
    
    def benchmark_algorithm(self, algorithm_func, *args, **kwargs):
        """Benchmark a single algorithm with multiple runs."""
        times = []
        num_runs = 3
        
        for run in range(num_runs):
            start_time = time.perf_counter()
            result = algorithm_func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'result_size': len(result) if hasattr(result, '__len__') else 0
        }
    
    def run_comparison(self, graph, sources, bound, graph_name="Unknown"):
        """Run a comprehensive comparison between BMSSP and Dijkstra."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {graph_name}")
        print(f"Graph: {len(graph)} nodes, Bound: {bound}, Sources: {len(sources)}")
        print(f"{'='*60}")
        
        # First verify correctness
        if not self.verify_correctness(graph, sources, bound):
            print("⚠️  Skipping performance benchmark due to correctness issues")
            return None
        
        print("\nRunning performance benchmarks...")
        
        # Benchmark Dijkstra
        print("  Benchmarking Dijkstra...")
        dijkstra_stats = self.benchmark_algorithm(
            dijkstra_bounded_multi_source, graph, sources, bound
        )
        
        # Benchmark BMSSP  
        print("  Benchmarking BMSSP...")
        def run_bmssp():
            solver = BMSSPSolver(graph)
            return solver.solve(sources, bound)
        
        bmssp_stats = self.benchmark_algorithm(run_bmssp)
        
        # Calculate speedup
        speedup = dijkstra_stats['mean_time'] / bmssp_stats['mean_time']
        
        # Report results
        print(f"\n📊 BENCHMARK RESULTS:")
        print(f"   Dijkstra: {dijkstra_stats['mean_time']:.6f}s (±{dijkstra_stats['max_time']-dijkstra_stats['min_time']:.6f}s)")
        print(f"   BMSSP:    {bmssp_stats['mean_time']:.6f}s (±{bmssp_stats['max_time']-bmssp_stats['min_time']:.6f}s)")
        print(f"   Speedup:  {'BMSSP' if speedup > 1 else 'Dijkstra'} is {abs(speedup):.2f}x faster")
        
        result = {
            'graph_name': graph_name,
            'num_nodes': len(graph),
            'num_sources': len(sources),
            'bound': bound,
            'dijkstra_time': dijkstra_stats['mean_time'],
            'bmssp_time': bmssp_stats['mean_time'],
            'speedup': speedup,
            'vertices_found': dijkstra_stats['result_size']
        }
        
        self.results.append(result)
        return result
    
    def print_summary(self):
        """Print a summary of all benchmark results."""
        if not self.results:
            print("No benchmark results to summarize.")
            return
        
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        dijkstra_wins = sum(1 for r in self.results if r['speedup'] < 1)
        bmssp_wins = len(self.results) - dijkstra_wins
        
        print(f"Total benchmarks: {len(self.results)}")
        print(f"Dijkstra wins: {dijkstra_wins}")
        print(f"BMSSP wins: {bmssp_wins}")
        
        if dijkstra_wins > 0:
            avg_dijkstra_advantage = statistics.mean([1/r['speedup'] for r in self.results if r['speedup'] < 1])
            print(f"Average Dijkstra advantage: {avg_dijkstra_advantage:.2f}x")
        
        if bmssp_wins > 0:
            avg_bmssp_advantage = statistics.mean([r['speedup'] for r in self.results if r['speedup'] > 1])
            print(f"Average BMSSP advantage: {avg_bmssp_advantage:.2f}x")


def main():
    """Main benchmark execution."""
    runner = BenchmarkRunner()
    
    # Test 1: Small connected graph
    print("Generating small connected graph...")
    small_graph = GraphGenerator.generate_sparse_connected(100, avg_degree=3)
    sources = random.sample(list(small_graph.keys()), 3)
    runner.run_comparison(small_graph, sources, bound=20, graph_name="Small Connected (100 nodes)")
    
    # Test 2: Medium sparse graph
    print("\nGenerating medium sparse graph...")
    medium_graph = GraphGenerator.generate_sparse_connected(500, avg_degree=4)
    sources = random.sample(list(medium_graph.keys()), 5)
    runner.run_comparison(medium_graph, sources, bound=30, graph_name="Medium Sparse (500 nodes)")
    
    # Test 3: Larger graph
    print("\nGenerating larger graph...")
    large_graph = GraphGenerator.generate_sparse_connected(1000, avg_degree=5)
    sources = random.sample(list(large_graph.keys()), 5)
    runner.run_comparison(large_graph, sources, bound=40, graph_name="Large Sparse (1000 nodes)")
    
    # Test 4: Grid graph (structured topology)
    print("\nGenerating 2D grid graph...")
    grid_graph = GraphGenerator.generate_grid_graph(20, 20)  # 400 nodes
    grid_nodes = list(grid_graph.keys())
    sources = random.sample(grid_nodes, 3)
    runner.run_comparison(grid_graph, sources, bound=25, graph_name="2D Grid (20x20)")
    
    # Test 5: Dense small graph
    print("\nGenerating dense small graph...")
    dense_graph = GraphGenerator.generate_erdos_renyi(200, edge_probability=0.3)
    sources = random.sample(list(dense_graph.keys()), 4)
    runner.run_comparison(dense_graph, sources, bound=15, graph_name="Dense Small (200 nodes)")
    
    # Final summary
    runner.print_summary()
    
    print(f"\n{'='*80}")
    print("ANALYSIS AND CONCLUSIONS")
    print(f"{'='*80}")
    
    if runner.results:
        avg_speedup = statistics.mean([r['speedup'] for r in runner.results])
        if avg_speedup < 1:
            print(f"Overall: Dijkstra is {1/avg_speedup:.2f}x faster on average")
            print("\nPossible reasons:")
            print("  • Constant factor overhead in BMSSP implementation")
            print("  • Python's recursion and object creation costs")
            print("  • Graph sizes too small to benefit from BMSSP's asymptotic advantages")
            print("  • Graph topologies don't favor BMSSP's divide-and-conquer approach")
        else:
            print(f"Overall: BMSSP is {avg_speedup:.2f}x faster on average")
            print("BMSSP shows promising results!")
        
        print(f"\nKey insights:")
        print(f"  • Tested graph sizes: {min(r['num_nodes'] for r in runner.results)} - {max(r['num_nodes'] for r in runner.results)} nodes")
        print(f"  • Average vertices found per test: {statistics.mean([r['vertices_found'] for r in runner.results]):.1f}")
        print(f"  • Both algorithms found identical results in all tests (correctness verified)")


if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    print("🚀 Starting Comprehensive BMSSP vs Dijkstra Benchmark")
    print("This may take a few minutes...")
    
    main()
    
    print("\n✅ Benchmark complete!")
    print("Check the results above for detailed performance analysis.")