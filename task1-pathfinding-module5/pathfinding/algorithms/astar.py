import heapq
from typing import Optional

from algorithms.base_search import SearchAlgorithm
from models.position import Position
from services.path_reconstructor import reconstruct_path


class AStar(SearchAlgorithm):
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self, maze, start, goal):
        open_set = [(self.heuristic(start, goal), 0, start)]
        parent: dict[Position, Optional[Position]] = {start: None}

        g_score = {start: 0}

        expanded = 0

        while open_set:
            f, g, current = heapq.heappop(open_set)
            expanded += 1

            if current == goal:
                return reconstruct_path(parent, goal), expanded

            for neighbor in maze.neighbors(current):
                tentative_g = g + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current

                    new_f = tentative_g + self.heuristic(neighbor, goal)

                    heapq.heappush(open_set, (new_f, tentative_g, neighbor))

        return None, expanded
