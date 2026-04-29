from collections import deque

from algorithms.base_search import SearchAlgorithm
from services.path_reconstructor import reconstruct_path


class BFS(SearchAlgorithm):
    def solve(self, maze, start, goal):
        queue = deque([start])
        visited = {start}
        parent = {start: None}
        expanded = 0

        while queue:
            current = queue.popleft()
            expanded += 1

            if current == goal:
                return reconstruct_path(parent, goal), expanded

            for neighbor in maze.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        return None, expanded
