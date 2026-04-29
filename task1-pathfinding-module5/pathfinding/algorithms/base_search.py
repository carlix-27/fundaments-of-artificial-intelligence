from abc import ABC, abstractmethod
from typing import Optional

from models.maze import Maze
from models.position import Position

PathResult = tuple[Optional[list[Position]], int]


class SearchAlgorithm(ABC):
    @abstractmethod
    def solve(self, maze: Maze, start: Position, goal: Position) -> PathResult:
        pass
