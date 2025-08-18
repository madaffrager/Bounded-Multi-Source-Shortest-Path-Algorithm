import heapq

def dijkstra_bounded(graph, sources, bound):
    """
    Standard bounded multi-source Dijkstra.
    Explore all nodes up to distance < bound.
    """
    pq = []
    dist = {v: float("inf") for v in graph}
    for s in sources:
        dist[s] = 0
        heapq.heappush(pq, (0, s))

    while pq:
        d, u = heapq.heappop(pq)
        if d >= bound:
            continue
        for v, w in graph[u].items():
            if d + w < bound and d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))

    return {v: d for v, d in dist.items() if d < bound}


# Demo
if __name__ == "__main__":
    test_graph = {
        'A': {'B': 1, 'E': 10},
        'B': {'C': 1, 'F': 8},
        'C': {'D': 1, 'G': 6},
        'D': {'H': 1},
        'E': {'F': 1, 'I': 5},
        'F': {'G': 1, 'J': 4},
        'G': {'H': 1, 'K': 3},
        'H': {'L': 1},
        'I': {'J': 1, 'M': 2},
        'J': {'K': 1, 'N': 2},
        'K': {'L': 1, 'O': 2},
        'L': {'P': 1},
        'M': {'N': 1},
        'N': {'O': 1},
        'O': {'P': 1},
        'P': {}
    }

    sources = ['A']
    bound = 15
    result = dijkstra_bounded(test_graph, sources, bound)

    print("Dijkstra results:")
    for v, d in sorted(result.items(), key=lambda x: x[1]):
        print(f"{v}: {d}")
