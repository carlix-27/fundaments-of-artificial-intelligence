from typing import Optional

from algorithms.base_search import SearchAlgorithm
from models.position import Position
from services.path_reconstructor import reconstruct_path


class DFS(SearchAlgorithm):
    def solve(self, maze, start, goal):
        stack = [start]
        visited = {start}
        parent: dict[Position, Optional[Position]] = {start: None}
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
