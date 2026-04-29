from models.maze import Maze
from models.position import Position


class Validator:
    @staticmethod
    def validate(
        maze: Maze, path: list[Position], start: Position, goal: Position
    ) -> tuple[bool, str]:

        if not path:
            return False, "Camino vacío"

        if path[0] != start:
            return False, "No inicia en start"

        if path[-1] != goal:
            return False, "No termina en goal"

        for i in range(len(path)):
            if not maze.is_valid(path[i]):
                return False, f"Celda inválida {path[i]}"

            if i > 0:
                r1, c1 = path[i - 1]
                r2, c2 = path[i]

                dr = abs(r1 - r2)
                dc = abs(c1 - c2)

                if dr + dc != 1:
                    return False, "Movimiento inválido"

        return True, "Camino válido"
