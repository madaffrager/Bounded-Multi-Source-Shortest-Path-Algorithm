# file: bmssp_algorithm_final.py

import heapq
import math

# --- Component 1: The Specialized Priority Queue (No Changes) ---
class SpecializedPriorityQueue:
    # ... (This class is complete and correct, no changes needed)
    def __init__(self):
        self.buckets = {}
        self.pending_updates = []
        self.min_bucket_idx = float('inf')
    def _get_bucket_idx(self, dist):
        if dist <= 0: return 0
        return math.floor(math.log2(dist))
    def INITIALIZE(self, M, B):
        max_idx = self._get_bucket_idx(B) if B > 0 else 0
        self.buckets = {i: [] for i in range(max_idx + 1)}
        self.pending_updates = []
        self.min_bucket_idx = float('inf')
    def BATCHDECREASEKEY(self, K):
        self.pending_updates.extend(K)
    def _process_pending_updates(self):
        if not self.pending_updates: return
        for vertex, dist in self.pending_updates:
            idx = self._get_bucket_idx(dist)
            if idx in self.buckets:
                self.buckets[idx].append((dist, vertex))
                if idx < self.min_bucket_idx:
                    self.min_bucket_idx = idx
        self.pending_updates = []
    def is_empty(self):
        self._process_pending_updates()
        return self.min_bucket_idx == float('inf')
    def EXTRACT_MIN(self):
        self._process_pending_updates()
        if self.min_bucket_idx == float('inf'):
            raise IndexError("EXTRACT_MIN from an empty queue")
        active_bucket_idx = self.min_bucket_idx
        active_bucket = self.buckets[active_bucket_idx]
        min_dist, min_vertex = min(active_bucket, key=lambda item: item[0])
        active_bucket.remove((min_dist, min_vertex))
        if not active_bucket:
            new_min_idx = float('inf')
            for i in range(active_bucket_idx + 1, len(self.buckets)):
                if self.buckets[i]:
                    new_min_idx = i
                    break
            self.min_bucket_idx = new_min_idx
        return active_bucket_idx, min_dist, min_vertex

# --- Component 2: The Main BMSSP Algorithm Class (Updated bmssp method) ---
class BMSSP:
    def __init__(self, graph):
        self.graph = graph
        # D is now specific to each call, so we don't initialize it here.
        # Global distances are needed to ensure we don't lose progress between recursive calls.
        self.global_distances = {vertex: float('inf') for vertex in self.graph}

    def _relax_edges(self, node_to_relax_from, current_bound):
        updates = []
        u = node_to_relax_from
        for v, weight in self.graph.get(u, {}).items():
            if self.global_distances[u] + weight < self.global_distances[v]:
                new_dist = self.global_distances[u] + weight
                if new_dist < current_bound:
                    self.global_distances[v] = new_dist
                    updates.append((v, new_dist))
        return updates

    def FastDijkstra(self, B, S):
        pq = []
        # FastDijkstra works with the global distances
        for source in S:
            if source in self.global_distances and self.global_distances[source] == float('inf'):
                self.global_distances[source] = 0
            # We still use a local distance tracker for the heap logic
            heapq.heappush(pq, (self.global_distances.get(source, 0), source))
        
        completed_vertices = {}
        while pq:
            dist, current_vertex = heapq.heappop(pq)
            if dist > self.global_distances[current_vertex] or dist >= B: continue
            completed_vertices[current_vertex] = dist
            for neighbor, weight in self.graph.get(current_vertex, {}).items():
                new_dist = dist + weight
                if new_dist < self.global_distances[neighbor]:
                    self.global_distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
        U_prime = set(completed_vertices.keys())
        return U_prime

    # --- THE FINAL RECURSIVE bmssp METHOD ---
    def bmssp(self, B, S, recursion_level=0):
        indent = "  " * recursion_level
        print(f"\n{indent}--- Starting BMSSP (Lvl {recursion_level}) with B={B:.2f} & S={S} ---")

        # The base case of the recursion is handled by FastDijkstra.
        # For small B, it effectively solves the problem.
        U = self.FastDijkstra(B, S)
        
        D = SpecializedPriorityQueue()
        D.INITIALIZE(M=None, B=B)
        
        # Seed the queue D with the initial frontier
        initial_batch = []
        for u in U:
            for v, weight in self.graph.get(u, {}).items():
                 if v not in U:
                    initial_batch.append((v, self.global_distances[v]))
        D.BATCHDECREASEKEY(initial_batch)

        print(f"{indent}Main loop starting...")
        k = 1
        
        while len(U) < k * (2**B) and not D.is_empty():
            try:
                # 1. DIVIDE: Pick a new subproblem to solve
                _, d_min, m_i = D.EXTRACT_MIN()
            except IndexError:
                break
            
            if m_i in U: continue

            # Define the subproblem's parameters
            B_i = B - d_min # The new bound is what's left
            S_i = {m_i}     # The new source is the node we extracted

            if B_i <= 0: continue

            # 2. CONQUER: Recursively solve the subproblem
            print(f"{indent}Recursively calling BMSSP for node '{m_i}' with new bound B'={B_i:.2f}")
            U_from_recursion = self.bmssp(B_i, S_i, recursion_level + 1)

            # 3. COMBINE: Integrate the results
            U.update(U_from_recursion)
            
            # And relax the edges from the newly discovered nodes
            new_updates = []
            for u_new in U_from_recursion:
                new_updates.extend(self._relax_edges(u_new, B))
            
            if new_updates:
                print(f"{indent}Recursion for '{m_i}' found {len(new_updates)} new paths. Batch updating D.")
                D.BATCHDECREASEKEY(new_updates)

        print(f"{indent}--- BMSSP (Lvl {recursion_level}) Completed. Returning {len(U)} nodes. ---")
        return U

# --- Demonstration ---
if __name__ == "__main__":
    sample_graph = {
        'A': {'B': 1, 'C': 8},
        'B': {'C': 6, 'D': 8},
        'C': {'D': 1, 'E': 5},
        'D': {'E': 1},
        'E': {},
        'F': {'G': 2},
        'G': {}
    }
    search_bound = 15
    source_nodes = {'A'}

    solver = BMSSP(sample_graph)
    # Initialize global distances for the first call
    for source in source_nodes:
        solver.global_distances[source] = 0

    found_vertices = solver.bmssp(B=search_bound, S=source_nodes)

    print(f"\nFinal Result: Vertices reachable from {source_nodes} within bound {search_bound} are:")
    final_distances = {v: solver.global_distances[v] for v in found_vertices}
    print(sorted(final_distances.items(), key=lambda item: item[1]))