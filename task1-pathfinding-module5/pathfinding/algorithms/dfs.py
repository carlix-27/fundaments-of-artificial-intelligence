from algorithms.base_search import SearchAlgorithm
from services.path_reconstructor import reconstruct_path


class DFS(SearchAlgorithm):
    def solve(self, maze, start, goal):
        stack = [start]
        visited = {start}
        parent = {start: None}
        expanded = 0

        while stack:
            current = stack.pop()
            expanded += 1

            if current == goal:
                return reconstruct_path(parent, goal), expanded

            for neighbor in maze.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    stack.append(neighbor)

        return None, expanded
