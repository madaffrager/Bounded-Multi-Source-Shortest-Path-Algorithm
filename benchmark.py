# file: benchmark.py

import random
import time
from bmssp_algorithm import BMSSP # Assuming you renamed the file
from baseline_dijkstra import dijkstra_bounded

def generate_sparse_graph(num_nodes, avg_degree):
    """Generates a random sparse graph."""
    print(f"Generating a sparse graph with {num_nodes} nodes...")
    graph = {i: {} for i in range(num_nodes)}
    for i in range(num_nodes):
        # Add a few edges to ensure connectivity (Erdos-Renyi style)
        num_edges = avg_degree
        for _ in range(num_edges):
            j = random.randint(0, num_nodes - 1)
            if i != j:
                weight = random.randint(1, 10)
                graph[i][j] = weight
    return graph

if __name__ == "__main__":
    # --- Parameters for the Benchmark ---
    NUM_NODES = 1000
    AVG_DEGREE = 4 # Average number of outgoing edges per node
    NUM_SOURCES = 5
    BOUND = 50

    # 1. Generate the graph
    graph = generate_sparse_graph(NUM_NODES, AVG_DEGREE)
    
    # 2. Select random source nodes
    sources = set(random.sample(range(NUM_NODES), NUM_SOURCES))

    print("\n--- Starting Benchmark ---")
    print(f"Graph Size: {NUM_NODES} nodes, Sources: {NUM_SOURCES}, Bound: {BOUND}\n")

    # 3. Benchmark our BMSSP Algorithm
    solver = BMSSP(graph)
    for source in sources: # Manually set initial distances for our solver
        solver.global_distances[source] = 0
    
    start_time_bmssp = time.time()
    result_bmssp = solver.bmssp(B=BOUND, S=sources)
    end_time_bmssp = time.time()
    duration_bmssp = end_time_bmssp - start_time_bmssp
    print(f"\nBMSSP finished in {duration_bmssp:.6f} seconds.")

    # 4. Benchmark the Baseline Dijkstra
    start_time_dijkstra = time.time()
    result_dijkstra = dijkstra_bounded(graph, sources, BOUND)
    end_time_dijkstra = time.time()
    duration_dijkstra = end_time_dijkstra - start_time_dijkstra
    print(f"Baseline Dijkstra finished in {duration_dijkstra:.6f} seconds.")

    # 5. Final Report
    print("\n--- Benchmark Results ---")
    print(f"BMSSP found {len(result_bmssp)} nodes.")
    print(f"Dijkstra found {len(result_dijkstra)} nodes.")
    if duration_bmssp < duration_dijkstra:
        print("🏆 BMSSP was faster!")
    else:
        print("Baseline Dijkstra was faster.")