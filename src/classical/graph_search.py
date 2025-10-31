"""
Graph search algorithms: BFS, DFS, and A*.

These are classic graph traversal and pathfinding algorithms used as baselines
for comparison with metaheuristic approaches on graph-based problems.

References
----------
.. [1] https://www.geeksforgeeks.org/dsa/breadth-first-search-or-bfs-for-a-graph/
.. [2] https://www.geeksforgeeks.org/dsa/depth-first-search-or-dfs-for-a-graph/
.. [3] https://en.wikipedia.org/wiki/A*_search_algorithm
"""

from collections import deque
import heapq
from typing import Dict, List, Tuple, Callable, Optional, Set


def bfs(graph: Dict[int, List[Tuple[int, float]]], start: int, goal: int) -> Optional[List[int]]:
    """
    Breadth-First Search (BFS) for finding shortest path in unweighted graphs.
    
    BFS explores nodes level by level, guaranteeing the shortest path in terms
    of number of edges (hops). It is complete and optimal for unweighted graphs.
    
    Parameters
    ----------
    graph : Dict[int, List[Tuple[int, float]]]
        Graph represented as adjacency list.
        graph[node] = [(neighbor1, cost1), (neighbor2, cost2), ...]
        For unweighted graphs, costs can be all 1.0.
    start : int
        Starting node.
    goal : int
        Goal/target node.
    
    Returns
    -------
    path : List[int] or None
        Shortest path from start to goal as list of nodes, or None if no path exists.
    
    Properties
    ----------
    - Complete: Yes (will find solution if one exists)
    - Optimal: Yes for unweighted graphs (finds path with minimum edges)
    - Time Complexity: O(|V| + |E|) where V is vertices, E is edges
    - Space Complexity: O(|V|)
    
    Examples
    --------
    >>> graph = {
    ...     0: [(1, 1), (2, 1)],
    ...     1: [(0, 1), (3, 1)],
    ...     2: [(0, 1), (3, 1)],
    ...     3: [(1, 1), (2, 1)]
    ... }
    >>> path = bfs(graph, start=0, goal=3)
    >>> print(path)  # [0, 1, 3] or [0, 2, 3]
    """
    if start == goal:
        return [start]
    
    # Queue for BFS: (node, path_to_node)
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        # Explore neighbors
        if current in graph:
            for neighbor, _ in graph[current]:
                if neighbor in visited:
                    continue
                
                new_path = path + [neighbor]
                
                # Check if goal reached
                if neighbor == goal:
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
    
    # No path found
    return None


def dfs(graph: Dict[int, List[Tuple[int, float]]], start: int, visited: Optional[Set[int]] = None) -> List[int]:
    """
    Depth-First Search (DFS) for graph traversal.
    
    DFS explores as far as possible along each branch before backtracking.
    Returns the order of visited nodes. Not guaranteed to find shortest path.
    
    Parameters
    ----------
    graph : Dict[int, List[Tuple[int, float]]]
        Graph represented as adjacency list.
        graph[node] = [(neighbor1, cost1), (neighbor2, cost2), ...]
    start : int
        Starting node for traversal.
    visited : Set[int] or None, optional
        Set of already visited nodes (for recursive calls).
    
    Returns
    -------
    traversal_order : List[int]
        List of nodes in the order they were visited.
    
    Properties
    ----------
    - Complete: No (can get stuck in infinite loops in cyclic graphs without visited tracking)
    - Optimal: No (does not guarantee shortest path)
    - Time Complexity: O(|V| + |E|)
    - Space Complexity: O(|V|) for recursion stack
    
    Examples
    --------
    >>> graph = {
    ...     0: [(1, 1), (2, 1)],
    ...     1: [(0, 1), (3, 1)],
    ...     2: [(0, 1), (3, 1)],
    ...     3: [(1, 1), (2, 1)]
    ... }
    >>> traversal = dfs(graph, start=0)
    >>> print(traversal)  # Example: [0, 1, 3, 2]
    """
    if visited is None:
        visited = set()
    
    # Mark current node as visited
    visited.add(start)
    traversal = [start]
    
    # Recursively visit neighbors
    if start in graph:
        for neighbor, _ in graph[start]:
            if neighbor not in visited:
                traversal.extend(dfs(graph, neighbor, visited))
    
    return traversal


def astar(
    graph: Dict[int, List[Tuple[int, float]]], 
    start: int, 
    goal: int, 
    heuristic: Callable[[int], float]
) -> Optional[Tuple[List[int], float]]:
    """
    A* search algorithm for optimal pathfinding with heuristic guidance.
    
    A* finds the optimal (lowest cost) path from start to goal using a heuristic
    to guide the search. The heuristic h(n) estimates the cost from node n to goal.
    
    A* is optimal if the heuristic is admissible (never overestimates true cost).
    
    Parameters
    ----------
    graph : Dict[int, List[Tuple[int, float]]]
        Graph represented as adjacency list with edge costs.
        graph[node] = [(neighbor1, cost1), (neighbor2, cost2), ...]
    start : int
        Starting node.
    goal : int
        Goal/target node.
    heuristic : Callable[[int], float]
        Heuristic function h(n) that estimates cost from node n to goal.
        Must be admissible (h(n) <= true_cost(n, goal)) for optimality.
    
    Returns
    -------
    result : Tuple[List[int], float] or None
        If path found: (path, total_cost) where path is list of nodes and
        total_cost is the actual path cost. Returns None if no path exists.
    
    Properties
    ----------
    - Complete: Yes (if heuristic is admissible and graph is finite)
    - Optimal: Yes (if heuristic is admissible: h(n) <= true_cost(n, goal))
    - Time Complexity: O(|E|) with good heuristic, O(b^d) worst case
    - Space Complexity: O(|V|)
    
    Notes
    -----
    The algorithm uses f(n) = g(n) + h(n) where:
    - g(n): actual cost from start to n
    - h(n): estimated cost from n to goal
    - f(n): estimated total cost of path through n
    
    Examples
    --------
    >>> # Grid graph example
    >>> graph = {
    ...     0: [(1, 1), (2, 1)],
    ...     1: [(0, 1), (3, 1)],
    ...     2: [(0, 1), (3, 1)],
    ...     3: [(1, 1), (2, 1)]
    ... }
    >>> # Simple heuristic (could be Euclidean distance in a grid)
    >>> def h(node):
    ...     # Distance to goal (node 3)
    ...     return abs(node - 3)
    >>> path, cost = astar(graph, start=0, goal=3, heuristic=h)
    >>> print(f"Path: {path}, Cost: {cost}")
    """
    if start == goal:
        return [start], 0.0
    
    # Priority queue: (f_score, node, g_score, path)
    open_set = [(heuristic(start), start, 0.0, [start])]
    visited = set()
    
    # Track best g_score for each node
    g_scores = {start: 0.0}
    
    while open_set:
        f_score, current, g_score, path = heapq.heappop(open_set)
        
        # Skip if already visited with better or equal cost
        if current in visited:
            continue
        
        visited.add(current)
        
        # Check if goal reached
        if current == goal:
            return path, g_score
        
        # Explore neighbors
        if current in graph:
            for neighbor, edge_cost in graph[current]:
                if neighbor in visited:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score + edge_cost
                
                # Only process if this is a better path to neighbor
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f, neighbor, tentative_g, new_path))
    
    # No path found
    return None


if __name__ == "__main__":
    print("=" * 60)
    print("GRAPH SEARCH ALGORITHMS DEMO")
    print("=" * 60)
    
    # Example graph (simple grid-like structure)
    #   0 -- 1
    #   |    |
    #   2 -- 3 -- 4
    #        |
    #        5
    
    graph = {
        0: [(1, 1.0), (2, 1.0)],
        1: [(0, 1.0), (3, 1.0)],
        2: [(0, 1.0), (3, 1.0)],
        3: [(1, 1.0), (2, 1.0), (4, 1.0), (5, 1.0)],
        4: [(3, 1.0)],
        5: [(3, 1.0)]
    }
    
    # Test BFS
    print("\n1. Breadth-First Search (BFS)")
    print("-" * 60)
    path_bfs = bfs(graph, start=0, goal=5)
    print(f"BFS path from 0 to 5: {path_bfs}")
    print(f"Path length: {len(path_bfs) - 1} edges")
    
    # Test DFS
    print("\n2. Depth-First Search (DFS)")
    print("-" * 60)
    traversal_dfs = dfs(graph, start=0)
    print(f"DFS traversal from 0: {traversal_dfs}")
    print(f"All nodes visited: {len(traversal_dfs)} nodes")
    
    # Test A*
    print("\n3. A* Search")
    print("-" * 60)
    
    # Define a simple heuristic (straight-line distance estimate)
    # In this example, we use a simple numerical distance to goal
    def heuristic_to_5(node: int) -> float:
        """Simple heuristic: numeric distance to node 5."""
        return abs(node - 5)
    
    result_astar = astar(graph, start=0, goal=5, heuristic=heuristic_to_5)
    
    if result_astar:
        path_astar, cost_astar = result_astar
        print(f"A* path from 0 to 5: {path_astar}")
        print(f"Total cost: {cost_astar}")
    else:
        print("No path found")
    
    # Test with weighted graph
    print("\n4. A* on Weighted Graph")
    print("-" * 60)
    
    weighted_graph = {
        0: [(1, 2.0), (2, 5.0)],
        1: [(0, 2.0), (3, 3.0)],
        2: [(0, 5.0), (3, 1.0)],
        3: [(1, 3.0), (2, 1.0), (4, 2.0)],
        4: [(3, 2.0)]
    }
    
    def heuristic_to_4(node: int) -> float:
        """Heuristic for reaching node 4."""
        return abs(node - 4)
    
    result_weighted = astar(weighted_graph, start=0, goal=4, heuristic=heuristic_to_4)
    
    if result_weighted:
        path_w, cost_w = result_weighted
        print(f"A* path from 0 to 4: {path_w}")
        print(f"Total cost: {cost_w}")
        print(f"Expected optimal: 0->2->3->4 with cost 8.0")
    
    # Test edge cases
    print("\n5. Edge Cases")
    print("-" * 60)
    
    # Start equals goal
    path_same = bfs(graph, start=3, goal=3)
    print(f"BFS same start/goal: {path_same}")
    
    # No path exists
    disconnected_graph = {
        0: [(1, 1.0)],
        1: [(0, 1.0)],
        2: [(3, 1.0)],
        3: [(2, 1.0)]
    }
    path_none = bfs(disconnected_graph, start=0, goal=3)
    print(f"BFS no path exists: {path_none}")
    
    print("\n" + "=" * 60)
    print("All graph search tests completed!")
    print("=" * 60)
    print("\nKey Properties:")
    print("- BFS: Complete, Optimal for unweighted graphs, O(V+E)")
    print("- DFS: Complete with cycle detection, Not optimal, O(V+E)")
    print("- A*: Complete & Optimal (with admissible heuristic), O(E) - O(b^d)")
