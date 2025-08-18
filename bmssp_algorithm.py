import heapq

class BMSSPSolver:
    """
    Clean implementation of the Bounded Multi-Source Shortest Path (BMSSP) algorithm.
    Based on recursive divide-and-conquer:
      1. Bounded exploration phase
      2. Frontier collection
      3. Recursive calls on frontier nodes
      4. Merge results
    """

    def __init__(self, graph):
        self.graph = graph
        self.vertices = set(graph.keys())
        for u in graph:
            for v in graph[u]:
                self.vertices.add(v)

        # global state
        self.global_distances = {}
        self.recursion_count = 0
        self.max_depth = 0

    def bounded_dijkstra(self, sources, bound):
        """
        Explore graph from sources, but stop at distance >= bound.
        Return visited nodes and frontier (nodes beyond the bound).
        """
        pq = []
        local_distances = {}
        visited = set()
        frontier = []

        for s in sources:
            start_dist = self.global_distances.get(s, 0)
            local_distances[s] = start_dist
            heapq.heappush(pq, (start_dist, s))

        while pq:
            dist, u = heapq.heappop(pq)
            if dist >= bound:
                continue
            if u in visited:
                continue
            visited.add(u)

            # update global distances
            if u not in self.global_distances or dist < self.global_distances[u]:
                self.global_distances[u] = dist

            for v, w in self.graph.get(u, {}).items():
                new_dist = dist + w
                if new_dist < bound:
                    if v not in visited:
                        if v not in local_distances or new_dist < local_distances[v]:
                            local_distances[v] = new_dist
                            heapq.heappush(pq, (new_dist, v))
                else:
                    # beyond current bound → frontier candidate
                    frontier.append((v, new_dist))

        return visited, frontier

    def bmssp(self, sources, bound, max_depth=20, depth=0):
        """
        Main recursive BMSSP.
        """
        self.recursion_count += 1
        self.max_depth = max(self.max_depth, depth)

        if bound <= 0 or depth >= max_depth:
            return set()

        # Phase 1: bounded exploration
        explored, frontier = self.bounded_dijkstra(sources, bound)

        # Phase 2: recurse on frontier nodes
        for v, d in frontier:
            remaining_bound = bound - (d - self.global_distances.get(v, float("inf")))
            if remaining_bound > 0:
                old_dist = self.global_distances.get(v, float("inf"))
                if d < old_dist:
                    self.global_distances[v] = d
                self.bmssp([v], bound - d, max_depth, depth + 1)


        return explored

    def solve(self, sources, bound):
        """
        Public interface to run BMSSP.
        """
        self.global_distances = {s: 0 for s in sources}
        self.recursion_count = 0
        self.max_depth = 0

        self.bmssp(sources, bound)

        return {v: d for v, d in self.global_distances.items() if d < bound}


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

    solver = BMSSPSolver(test_graph)
    sources = ['A']
    bound = 15

    result = solver.solve(sources, bound)

    print("BMSSP results:")
    for v, d in sorted(result.items(), key=lambda x: x[1]):
        print(f"{v}: {d}")
    print(f"Recursive calls: {solver.recursion_count}, Max depth: {solver.max_depth}")
