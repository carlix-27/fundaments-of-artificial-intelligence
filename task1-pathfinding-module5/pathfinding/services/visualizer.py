from models.maze import Maze


class Visualizer:
    @staticmethod
    def show(maze: Maze, path=None):

        path = path or []
        path_set = set(path)

        for r in range(maze.rows):
            row = ""

            for c in range(maze.cols):
                pos = (r, c)

                if pos == (0, 0):
                    row += "S "

                elif pos == (maze.rows - 1, maze.cols - 1):
                    row += "G "

                elif pos in path_set:
                    row += "· "

                elif maze.grid[r][c] == 1:
                    row += "█ "

                else:
                    row += ". "

            print(row)
