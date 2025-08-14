import heapq
from collections import defaultdict

def dijkstra_bounded_multi_source(graph, sources, bound):
    """
    Multi-source bounded Dijkstra's algorithm.
    
    Args:
        graph: Dictionary representing adjacency list {u: {v: weight, ...}, ...}
        sources: List or set of source vertices
        bound: Maximum path length to consider
    
    Returns:
        Dictionary mapping reachable vertices to their shortest distances
    """
    # Initialize distances
    all_vertices = set()
    for u in graph:
        all_vertices.add(u)
        for v in graph[u]:
            all_vertices.add(v)
    
    distances = {v: float('inf') for v in all_vertices}
    
    # Priority queue: (distance, vertex)
    pq = []
    
    # Initialize sources
    for source in sources:
        if source in distances:
            distances[source] = 0
            heapq.heappush(pq, (0, source))
    
    visited = set()
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        # Skip if we've already processed this vertex with a better distance
        if current_dist > distances[u]:
            continue
        
        # Skip if distance exceeds bound
        if current_dist >= bound:
            continue
            
        # Mark as visited
        visited.add(u)
        
        # Relax all outgoing edges
        for v, weight in graph.get(u, {}).items():
            new_dist = current_dist + weight
            
            # Only consider if within bound and improves current distance
            if new_dist < bound and new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    
    # Return only reachable vertices with their distances
    return {v: distances[v] for v in visited if distances[v] < bound}


def dijkstra_single_source(graph, source, bound=float('inf')):
    """
    Single-source Dijkstra's algorithm with optional bound.
    
    Args:
        graph: Dictionary representing adjacency list
        source: Starting vertex
        bound: Maximum path length (optional)
    
    Returns:
        Dictionary mapping vertices to shortest distances from source
    """
    return dijkstra_bounded_multi_source(graph, [source], bound)


# Testing and validation
if __name__ == "__main__":
    # Test graph
    test_graph = {
        'A': {'B': 1, 'C': 8},
        'B': {'C': 2, 'D': 3},
        'C': {'D': 1, 'E': 5},
        'D': {'E': 1, 'F': 4},
        'E': {'F': 2},
        'F': {'G': 1},
        'G': {}
    }
    
    print("Testing Baseline Dijkstra Implementation")
    print("=" * 45)
    
    # Test single source
    print("\nSingle-source from 'A' with bound 10:")
    result_single = dijkstra_single_source(test_graph, 'A', 10)
    for vertex, dist in sorted(result_single.items(), key=lambda x: x[1]):
        print(f"  {vertex}: {dist}")
    
    # Test multi-source
    print("\nMulti-source from ['A', 'F'] with bound 8:")
    result_multi = dijkstra_bounded_multi_source(test_graph, ['A', 'F'], 8)
    for vertex, dist in sorted(result_multi.items(), key=lambda x: x[1]):
        print(f"  {vertex}: {dist}")