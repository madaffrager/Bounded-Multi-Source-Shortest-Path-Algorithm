import heapq
import math
from collections import defaultdict, deque

class SpecializedPriorityQueue:
    """
    A bucket-based priority queue optimized for batch decrease-key operations.
    Uses logarithmic bucketing to group nodes by distance ranges.
    """
    
    def __init__(self):
        self.buckets = {}
        self.pending_updates = []
        self.min_bucket_idx = float('inf')
        self.size = 0
    
    def _get_bucket_idx(self, dist):
        """Map distance to bucket index using logarithmic bucketing."""
        if dist <= 0:
            return 0
        if dist == 1:
            return 0
        return math.floor(math.log2(dist))
    
    def initialize(self, max_bound):
        """Initialize buckets for distances up to max_bound."""
        if max_bound <= 0:
            max_bucket = 0
        else:
            max_bucket = self._get_bucket_idx(max_bound)
        
        self.buckets = {i: [] for i in range(max_bucket + 1)}
        self.pending_updates = []
        self.min_bucket_idx = float('inf')
        self.size = 0
    
    def batch_decrease_key(self, updates):
        """Add multiple (vertex, distance) updates to be processed lazily."""
        self.pending_updates.extend(updates)
    
    def _process_pending_updates(self):
        """Process all pending updates and update bucket structure."""
        if not self.pending_updates:
            return
            
        for vertex, dist in self.pending_updates:
            bucket_idx = self._get_bucket_idx(dist)
            if bucket_idx in self.buckets:
                self.buckets[bucket_idx].append((dist, vertex))
                self.size += 1
                if bucket_idx < self.min_bucket_idx:
                    self.min_bucket_idx = bucket_idx
        
        self.pending_updates = []
    
    def is_empty(self):
        """Check if queue is empty after processing pending updates."""
        self._process_pending_updates()
        return self.size == 0
    
    def extract_min(self):
        """Extract the minimum distance vertex from the queue."""
        self._process_pending_updates()
        
        if self.size == 0:
            raise IndexError("extract_min from empty queue")
        
        # Find the minimum non-empty bucket
        while self.min_bucket_idx < len(self.buckets) and not self.buckets[self.min_bucket_idx]:
            self.min_bucket_idx += 1
        
        if self.min_bucket_idx >= len(self.buckets):
            raise IndexError("No non-empty buckets found")
        
        # Extract minimum from this bucket
        bucket = self.buckets[self.min_bucket_idx]
        min_item = min(bucket, key=lambda x: x[0])
        bucket.remove(min_item)
        self.size -= 1
        
        # Update min_bucket_idx if this bucket is now empty
        if not bucket:
            self._find_next_min_bucket()
        
        return min_item[1], min_item[0]  # vertex, distance
    
    def _find_next_min_bucket(self):
        """Find the next non-empty bucket index."""
        self.min_bucket_idx = float('inf')
        for idx in sorted(self.buckets.keys()):
            if self.buckets[idx]:
                self.min_bucket_idx = idx
                break


class BMSSPSolver:
    """
    Implementation of the Bounded Multi-Source Shortest Path algorithm.
    """
    
    def __init__(self, graph):
        self.graph = graph
        self.vertices = set()
        for u in graph:
            self.vertices.add(u)
            for v in graph[u]:
                self.vertices.add(v)
        
        # Global state shared across recursive calls
        self.global_distances = {v: float('inf') for v in self.vertices}
        self.visited_global = set()
    
    def fast_dijkstra(self, sources, bound):
        """
        A bounded Dijkstra's algorithm that serves as the base case
        for BMSSP recursion. Explores easily reachable nodes first.
        """
        pq = []
        local_visited = set()
        
        # Initialize with sources
        for source in sources:
            if source in self.global_distances:
                if self.global_distances[source] == float('inf'):
                    self.global_distances[source] = 0
                heapq.heappush(pq, (self.global_distances[source], source))
        
        while pq:
            dist, u = heapq.heappop(pq)
            
            # Skip if we've found a better path or exceeded bound
            if u in local_visited or dist >= bound:
                continue
            
            local_visited.add(u)
            
            # Relax edges
            for v, weight in self.graph.get(u, {}).items():
                new_dist = dist + weight
                if new_dist < self.global_distances[v] and new_dist < bound:
                    self.global_distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
        
        return local_visited
    
    def _collect_frontier(self, explored_set, bound):
        """
        Collect frontier nodes (unexplored nodes adjacent to explored ones)
        for seeding the specialized priority queue.
        """
        frontier = []
        for u in explored_set:
            for v, weight in self.graph.get(u, {}).items():
                if v not in explored_set and self.global_distances[v] < bound:
                    frontier.append((v, self.global_distances[v]))
        return frontier
    
    def bmssp(self, sources, bound, max_recursion_depth=10, current_depth=0):
        """
        Main BMSSP algorithm implementation.
        
        Args:
            sources: Set of source vertices
            bound: Maximum path length to consider
            max_recursion_depth: Prevent infinite recursion
            current_depth: Current recursion level
        
        Returns:
            Set of vertices reachable within the bound
        """
        indent = "  " * current_depth
        print(f"{indent}BMSSP (depth {current_depth}): bound={bound:.2f}, sources={len(sources)}")
        
        # Base case: Use fast Dijkstra for initial exploration
        explored = self.fast_dijkstra(sources, bound)
        
        # Termination conditions
        if (current_depth >= max_recursion_depth or 
            bound <= 1 or 
            len(explored) == 0):
            return explored
        
        # Initialize specialized priority queue
        pq = SpecializedPriorityQueue()
        pq.initialize(bound)
        
        # Seed with frontier nodes
        frontier = self._collect_frontier(explored, bound)
        if frontier:
            pq.batch_decrease_key(frontier)
        
        # Main recursive exploration loop
        iteration = 0
        max_iterations = min(100, len(self.vertices))  # Prevent infinite loops
        
        while not pq.is_empty() and iteration < max_iterations:
            try:
                vertex, distance = pq.extract_min()
            except IndexError:
                break
            
            # Skip if already explored or distance exceeds bound
            if vertex in explored or distance >= bound:
                iteration += 1
                continue
            
            # Define recursive subproblem
            remaining_bound = bound - distance
            if remaining_bound <= 0:
                iteration += 1
                continue
            
            print(f"{indent}  Recursing on vertex {vertex} with remaining bound {remaining_bound:.2f}")
            
            # Recursive call with tighter bound
            sub_explored = self.bmssp({vertex}, remaining_bound, max_recursion_depth, current_depth + 1)
            
            # Merge results
            new_vertices = sub_explored - explored
            explored.update(sub_explored)
            
            # Update frontier with newly accessible vertices
            if new_vertices:
                new_frontier = self._collect_frontier(new_vertices, bound)
                if new_frontier:
                    pq.batch_decrease_key(new_frontier)
            
            iteration += 1
        
        print(f"{indent}BMSSP (depth {current_depth}) found {len(explored)} vertices")
        return explored
    
    def solve(self, sources, bound):
        """
        Public interface for solving BMSSP.
        
        Args:
            sources: Set or list of source vertices
            bound: Maximum path length
            
        Returns:
            Dictionary mapping vertices to their shortest distances
        """
        # Reset global state
        self.global_distances = {v: float('inf') for v in self.vertices}
        self.visited_global = set()
        
        # Set source distances
        sources = set(sources)
        for source in sources:
            if source in self.global_distances:
                self.global_distances[source] = 0
        
        # Run algorithm
        reachable = self.bmssp(sources, bound)
        
        # Return results
        return {v: self.global_distances[v] for v in reachable 
                if self.global_distances[v] < bound}


# Demonstration
if __name__ == "__main__":
    # Test graph
    sample_graph = {
        'A': {'B': 1, 'C': 8},
        'B': {'C': 2, 'D': 3},
        'C': {'D': 1, 'E': 5},
        'D': {'E': 1, 'F': 4},
        'E': {'F': 2},
        'F': {'G': 1},
        'G': {}
    }
    
    print("Testing BMSSP Implementation")
    print("=" * 40)
    
    solver = BMSSPSolver(sample_graph)
    sources = ['A']
    bound = 10
    
    result = solver.solve(sources, bound)
    
    print(f"\nFinal Results:")
    print(f"Sources: {sources}")
    print(f"Bound: {bound}")
    print(f"Reachable vertices and distances:")
    
    for vertex, distance in sorted(result.items(), key=lambda x: x[1]):
        print(f"  {vertex}: {distance}")