# file: baseline_dijkstra.py

import heapq

def dijkstra_bounded(graph, sources, bound):
    """
    A standard multi-source Dijkstra's algorithm that stops
    exploring paths longer than a given bound.
    """
    distances = {node: float('inf') for node in graph}
    pq = []

    for source in sources:
        if source in distances:
            distances[source] = 0
            heapq.heappush(pq, (0, source))

    visited = set()

    while pq:
        dist, u = heapq.heappop(pq)

        # If we found a shorter path already, skip
        if dist > distances[u]:
            continue
        
        visited.add(u)

        # Explore neighbors
        for v, weight in graph.get(u, {}).items():
            new_dist = dist + weight
            # Only consider paths within the bound
            if new_dist < distances[v] and new_dist < bound:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
                
    return visited