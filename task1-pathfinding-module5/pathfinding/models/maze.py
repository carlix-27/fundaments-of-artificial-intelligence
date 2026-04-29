from typing import List

from models.position import Position


class Maze:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def is_valid(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0

    def neighbors(self, pos: Position):
        r, c = pos

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        result = []

        for dr, dc in directions:
            nxt = (r + dr, c + dc)
            if self.is_valid(nxt):
                result.append(nxt)

        return result
