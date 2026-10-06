"""
Road Network Optimization using Minimum Spanning Tree (MST)

Problem:
A government wants to connect several villages with roads.
The cost of constructing a road between two villages is different for each pair.
Every village must be connected to the road network while keeping total construction cost minimal.

Solutions provided:
1. Kruskal's Algorithm (using Disjoint Set Union / Union-Find) - O(E log E)
2. Prim's Algorithm (using Min-Heap / Priority Queue) - O(E log V)
"""

from typing import List, Tuple, Dict, Any
import heapq


# ==========================================
# 1. Kruskal's Algorithm (Union-Find / DSU)
# ==========================================

class DisjointSetUnion:
    """Disjoint Set Union (DSU) with Path Compression and Union by Rank."""
    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  # Path compression
        return self.parent[item]

    def union(self, item1, item2) -> bool:
        root1 = self.find(item1)
        root2 = self.find(item2)

        if root1 == root2:
            return False  # Already in the same component, would form a cycle

        # Union by rank
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1

        return True


def kruskal_mst(villages: List[str], roads: List[Tuple[str, str, float]]) -> Tuple[List[Tuple[str, str, float]], float]:
    """
    Finds the Minimum Spanning Tree using Kruskal's algorithm.

    :param villages: List of village names/identifiers.
    :param roads: List of candidate roads as (village_u, village_v, cost).
    :return: (selected_roads, total_cost)
    """
    # Sort all roads in ascending order of construction cost
    sorted_roads = sorted(roads, key=lambda edge: edge[2])
    dsu = DisjointSetUnion(villages)

    selected_roads = []
    total_cost = 0.0

    for u, v, cost in sorted_roads:
        # If adding edge (u, v) does not create a cycle
        if dsu.union(u, v):
            selected_roads.append((u, v, cost))
            total_cost += cost
            # An MST on V vertices always has exactly V - 1 edges
            if len(selected_roads) == len(villages) - 1:
                break

    if len(selected_roads) != len(villages) - 1 and len(villages) > 1:
        raise ValueError("The village network is disconnected; cannot connect all villages.")

    return selected_roads, total_cost


# ==========================================
# 2. Prim's Algorithm (Min-Heap / Priority Queue)
# ==========================================

def prim_mst(villages: List[str], roads: List[Tuple[str, str, float]]) -> Tuple[List[Tuple[str, str, float]], float]:
    """
    Finds the Minimum Spanning Tree using Prim's algorithm.

    :param villages: List of village names/identifiers.
    :param roads: List of candidate roads as (village_u, village_v, cost).
    :return: (selected_roads, total_cost)
    """
    if not villages:
        return [], 0.0

    # Build adjacency list
    adj: Dict[str, List[Tuple[float, str]]] = {v: [] for v in villages}
    for u, v, cost in roads:
        adj[u].append((cost, v))
        adj[v].append((cost, u))

    visited = set()
    selected_roads = []
    total_cost = 0.0

    # Min-heap stores: (cost, from_village, to_village)
    min_heap: List[Tuple[float, str, str]] = []

    # Start from an arbitrary village
    start_village = villages[0]
    visited.add(start_village)

    for cost, neighbor in adj[start_village]:
        heapq.heappush(min_heap, (cost, start_village, neighbor))

    while min_heap and len(visited) < len(villages):
        cost, u, v = heapq.heappop(min_heap)

        if v in visited:
            continue

        visited.add(v)
        selected_roads.append((u, v, cost))
        total_cost += cost

        for next_cost, next_neighbor in adj[v]:
            if next_neighbor not in visited:
                heapq.heappush(min_heap, (next_cost, v, next_neighbor))

    if len(visited) != len(villages):
        raise ValueError("The village network is disconnected; cannot connect all villages.")

    return selected_roads, total_cost


# ==========================================
# Demo / Example Execution
# ==========================================

if __name__ == "__main__":
    # Example village list
    villages = ["Village A", "Village B", "Village C", "Village D", "Village E"]

    # Candidate roads: (Village 1, Village 2, Construction Cost)
    roads = [
        ("Village A", "Village B", 4.0),
        ("Village A", "Village C", 2.0),
        ("Village B", "Village C", 1.0),
        ("Village B", "Village D", 5.0),
        ("Village C", "Village D", 8.0),
        ("Village C", "Village E", 10.0),
        ("Village D", "Village E", 2.0),
        ("Village D", "Village A", 6.0),
    ]

    print("=" * 60)
    print("VILLAGE ROAD NETWORK - MINIMUM SPANNING TREE (MST)")
    print("=" * 60)

    print("\nRunning Kruskal's Algorithm...")
    kruskal_selected, kruskal_cost = kruskal_mst(villages, roads)
    print(f"Total Minimum Cost: {kruskal_cost}")
    print("Constructed Roads:")
    for u, v, cost in kruskal_selected:
        print(f"  - Connect {u} <---> {v} (Cost: {cost})")

    print("\nRunning Prim's Algorithm...")
    prim_selected, prim_cost = prim_mst(villages, roads)
    print(f"Total Minimum Cost: {prim_cost}")
    print("Constructed Roads:")
    for u, v, cost in prim_selected:
        print(f"  - Connect {u} <---> {v} (Cost: {cost})")
