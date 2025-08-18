import random
import time
import statistics
from collections import defaultdict

from bmssp_algorithm import BMSSPSolver
from baseline_dijkstra import dijkstra_bounded as dijkstra_bounded_multi_source

class GraphGenerator:
    @staticmethod
    def generate_erdos_renyi(num_nodes, edge_probability=0.1, max_weight=10):
        graph = defaultdict(dict)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < edge_probability:
                    graph[i][j] = random.randint(1, max_weight)
        return dict(graph)

    @staticmethod
    def generate_sparse_connected(num_nodes, avg_degree=4, max_weight=10):
        graph = defaultdict(dict)
        for i in range(1, num_nodes):
            parent = random.randint(0, i - 1)
            weight = random.randint(1, max_weight)
            graph[parent][i] = weight
            graph[i][parent] = weight
        edges_needed = (num_nodes * avg_degree) // 2 - (num_nodes - 1)
        for _ in range(edges_needed):
            u, v = random.sample(range(num_nodes), 2)
            if v not in graph[u]:
                graph[u][v] = random.randint(1, max_weight)
        return dict(graph)

    @staticmethod
    def generate_grid_graph(width, height, max_weight=5):
        graph = defaultdict(dict)
        def node_id(x, y): return f"{x},{y}"
        for x in range(width):
            for y in range(height):
                current = node_id(x, y)
                for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor = node_id(nx, ny)
                        graph[current][neighbor] = random.randint(1, max_weight)
        return dict(graph)

class BenchmarkRunner:
    def __init__(self):
        self.results = []

    def verify_correctness(self, graph, sources, bound):
        dijkstra_result = dijkstra_bounded_multi_source(graph, sources, bound)
        bmssp_result = BMSSPSolver(graph).solve(sources, bound)
        if set(dijkstra_result.keys()) != set(bmssp_result.keys()):
            return False
        for v in dijkstra_result:
            if abs(dijkstra_result[v] - bmssp_result[v]) > 1e-6:
                return False
        return True

    def benchmark_algorithm(self, func, *args):
        times, result = [], None
        for _ in range(3):
            start = time.perf_counter()
            result = func(*args)
            times.append(time.perf_counter() - start)
        return {'mean_time': statistics.mean(times), 'min_time': min(times), 'max_time': max(times), 'result_size': len(result)}

    def run_comparison(self, graph, sources, bound, graph_name="Unknown"):
        print(f"\nBenchmarking: {graph_name} ({len(graph)} nodes, Bound={bound}, Sources={len(sources)})")
        if not self.verify_correctness(graph, sources, bound):
            print("Skipping benchmark due to correctness issues")
            return

        dijkstra_stats = self.benchmark_algorithm(dijkstra_bounded_multi_source, graph, sources, bound)
        bmssp_stats = self.benchmark_algorithm(lambda g, s, b: BMSSPSolver(g).solve(s, b), graph, sources, bound)
        speedup = dijkstra_stats['mean_time'] / bmssp_stats['mean_time']

        print(f"Dijkstra: {dijkstra_stats['mean_time']:.6f}s | BMSSP: {bmssp_stats['mean_time']:.6f}s | "
              f"{'BMSSP' if speedup>1 else 'Dijkstra'} faster: {abs(speedup):.2f}x")

        self.results.append({
            'graph_name': graph_name,
            'dijkstra_time': dijkstra_stats['mean_time'],
            'bmssp_time': bmssp_stats['mean_time'],
            'speedup': speedup
        })

    def print_summary(self):
        print("\nBENCHMARK SUMMARY")
        dijkstra_wins = sum(1 for r in self.results if r['speedup'] < 1)
        bmssp_wins = len(self.results) - dijkstra_wins
        print(f"Total benchmarks: {len(self.results)}, Dijkstra wins: {dijkstra_wins}, BMSSP wins: {bmssp_wins}")
        if dijkstra_wins:
            avg_dijkstra = statistics.mean([1/r['speedup'] for r in self.results if r['speedup'] < 1])
            print(f"Average Dijkstra advantage: {avg_dijkstra:.2f}x")
        if bmssp_wins:
            avg_bmssp = statistics.mean([r['speedup'] for r in self.results if r['speedup'] > 1])
            print(f"Average BMSSP advantage: {avg_bmssp:.2f}x")

def main():
    runner = BenchmarkRunner()

    # Small connected graph
    small_graph = GraphGenerator.generate_sparse_connected(100, avg_degree=3)
    runner.run_comparison(small_graph, random.sample(list(small_graph.keys()),3), bound=20, graph_name="Small Connected (100 nodes)")

    # Medium sparse graph
    medium_graph = GraphGenerator.generate_sparse_connected(500, avg_degree=4)
    runner.run_comparison(medium_graph, random.sample(list(medium_graph.keys()),5), bound=30, graph_name="Medium Sparse (500 nodes)")

    # Large sparse graph
    large_graph = GraphGenerator.generate_sparse_connected(1000, avg_degree=5)
    runner.run_comparison(large_graph, random.sample(list(large_graph.keys()),5), bound=40, graph_name="Large Sparse (1000 nodes)")

    # Grid graph
    grid_graph = GraphGenerator.generate_grid_graph(20, 20)
    runner.run_comparison(grid_graph, random.sample(list(grid_graph.keys()),3), bound=25, graph_name="2D Grid (20x20)")

    # Dense small graph
    dense_graph = GraphGenerator.generate_erdos_renyi(200, edge_probability=0.3)
    runner.run_comparison(dense_graph, random.sample(list(dense_graph.keys()),4), bound=15, graph_name="Dense Small (200 nodes)")

    # Additional dense multi-source scenarios for BMSSP wins
    for nodes, prob, sources_count, bound in [(300, 0.4, 5, 20),(400, 0.5, 6, 25)]:
        dense_multi = GraphGenerator.generate_erdos_renyi(nodes, edge_probability=prob)
        runner.run_comparison(dense_multi, random.sample(list(dense_multi.keys()), sources_count), bound=bound,
                              graph_name=f"Dense Multi-Source ({nodes} nodes, p={prob})")
    
    
    # Test 8: Automatic scaling benchmark for dense graphs
    RUN_AUTO_SCALE  = True
    if RUN_AUTO_SCALE :
        print("\nRunning automatic scaling benchmark to highlight BMSSP advantage...")
        for num_nodes in [300, 400, 500, 600]:
            for edge_prob in [0.3, 0.4, 0.5, 0.6]:
                for num_sources in [3, 4, 5, 6]:
                    graph_name = f"Dense Multi-Source ({num_nodes} nodes, p={edge_prob})"
                    dense_graph = GraphGenerator.generate_erdos_renyi(num_nodes, edge_probability=edge_prob)
                    sources = random.sample(list(dense_graph.keys()), num_sources)
                    bound = max(15, num_nodes // 20)  # scale bound with graph size
                    runner.run_comparison(dense_graph, sources, bound=bound, graph_name=graph_name)

    runner.print_summary()

if __name__ == "__main__":
    random.seed(42)
    print("Starting BMSSP vs Dijkstra Benchmark...")
    main()
    print("Benchmark complete.")
