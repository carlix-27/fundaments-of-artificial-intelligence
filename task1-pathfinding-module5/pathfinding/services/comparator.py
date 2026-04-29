class Comparator:
    @staticmethod
    def run(maze, algorithms, start, goal):

        for algo in algorithms:
            path, expanded = algo.solve(maze, start, goal)

            print(algo.__class__.__name__)
            print("Expanded:", expanded)
            print("Length:", len(path))
            print()
