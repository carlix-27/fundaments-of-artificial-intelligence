from algorithms.astar import AStar
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from data.mazes import MAZE_15
from models.maze import Maze
from services.comparator import Comparator

maze = Maze(MAZE_15)

algorithms = [BFS(), DFS(), AStar()]

Comparator.run(maze, algorithms, (0, 0), (14, 14))
