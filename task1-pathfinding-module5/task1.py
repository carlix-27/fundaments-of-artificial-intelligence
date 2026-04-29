# %% [markdown]
# # Práctica Bloque 5: Pathfinding en Laberintos
#
# **Curso:** Fundamentos e Historia de la Inteligencia Artificial
# **Universidad Austral**
#
# ## Objetivos
# - Implementar BFS, DFS y A* a partir del pseudocódigo y el starter provisto
# - Comparar su eficiencia (nodos expandidos)
# - Entender el poder de las heurísticas
#
# ---

# %% [markdown]
# ## Idea clave
#
# Todos los algoritmos que vas a implementar comparten la misma estructura:
#
# while frontera:
#     tomar nodo actual
#     si es objetivo -> terminar
#     expandir vecinos
#
# 👉 Lo único que cambia es cómo se elige el próximo nodo:
# - BFS: FIFO (cola)
# - DFS: LIFO (pila)
# - A*: menor f(n) = g(n) + h(n)
#
# %% [markdown]
# ## 1. Setup y Laberintos

# %%
import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

# Laberinto 10x10
MAZE_10 = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
]

# Laberinto 15x15
MAZE_15 = [
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# Laberinto 20x20
MAZE_20 = [
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
]

# Laberinto 25x25
MAZE_25_EXTRA = [
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
]

# Tipo para posiciones
Position = Tuple[int, int]

# %% [markdown]
# ## 2. Función auxiliar: obtener vecinos


# %%
def get_neighbors(maze: List[List[int]], pos: Position) -> List[Position]:
    """
    Devuelve los vecinos válidos de una posición (4 direcciones).

    Args:
        maze: Matriz del laberinto (0=libre, 1=pared)
        pos: Tupla (fila, columna) de la posición actual

    Returns:
        Lista de posiciones vecinas transitables
    """
    r, c = pos
    rows, cols = len(maze), len(maze[0])
    neighbors = []

    # 4 direcciones: Norte, Sur, Oeste, Este
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        # Verificar límites y que no sea pared
        if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
            neighbors.append((nr, nc))

    return neighbors


# Prueba rápida
print("Vecinos de (0,0) en MAZE_15:", get_neighbors(MAZE_15, (0, 0)))
print("Vecinos de (6,6) en MAZE_15:", get_neighbors(MAZE_15, (6, 6)))

# %% [markdown]
# ## 3. BFS - Breadth-First Search
#
# **Tu tarea:** Completa la implementación de BFS.
#
# Recuerda:
# - Usa una **cola (FIFO)** - `deque` con `append()` y `popleft()`
# - Marca los nodos como visitados **cuando los agregas a la cola**
# - Cuenta los nodos **expandidos** (cada vez que sacas uno de la cola)


# %%
def bfs(
    maze: List[List[int]], start: Position, goal: Position
) -> Tuple[Optional[List[Position]], int]:
    """
    Breadth-First Search.

    Args:
        maze: Matriz del laberinto
        start: Posición inicial (fila, columna)
        goal: Posición objetivo (fila, columna)

    Returns:
        Tupla (camino, nodos_expandidos)
        - camino: Lista de posiciones desde start hasta goal, o None si no existe
        - nodos_expandidos: Cantidad de nodos sacados de la frontera
    """
    # TODO: Implementa BFS

    # Inicializar la cola (frontera)
    queue = deque([start])

    # Conjunto de visitados (para evitar ciclos)
    visited = {start}

    # Diccionario para reconstruir el camino
    parent: Dict[Position, Optional[Position]] = {start: None}

    # Contador de nodos expandidos
    expanded = 0

    while queue:
        # TODO: Sacar el siguiente nodo de la cola (FIFO)
        # Pista: deque usa popleft()
        current = None  # <-- COMPLETAR
        expanded += 1

        # TODO: Verificar si llegamos al objetivo
        # Si current == goal, reconstruir y retornar el camino

        # TODO: Expandir vecinos
        # Para cada vecino no visitado:
        #   - Marcarlo como visitado
        #   - Guardar su parent
        #   - Agregarlo a la cola

        pass  # <-- ELIMINAR cuando implementes

    return None, expanded  # No se encontró camino


# %% [markdown]
# ## 4. DFS - Depth-First Search
#
# **Tu tarea:** Completa la implementación de DFS.
#
# Recuerda:
# - Usa una **pila (LIFO)** - lista con `append()` y `pop()`
# - También necesitas `visited` para evitar ciclos infinitos


# %%
def dfs(
    maze: List[List[int]], start: Position, goal: Position
) -> Tuple[Optional[List[Position]], int]:
    """
    Depth-First Search.

    Args:
        maze: Matriz del laberinto
        start: Posición inicial
        goal: Posición objetivo

    Returns:
        Tupla (camino, nodos_expandidos)
    """
    # TODO: Implementa DFS

    # Inicializar la pila
    stack = [start]

    # Visitados y parents
    visited = {start}
    parent: Dict[Position, Optional[Position]] = {start: None}

    expanded = 0

    while stack:
        # TODO: Sacar de la pila (LIFO)
        # Pista: list usa pop()
        current = None  # <-- COMPLETAR
        expanded += 1

        # TODO: Verificar objetivo

        # TODO: Expandir vecinos

        pass  # <-- ELIMINAR

    return None, expanded


# %% [markdown]
# ## 5. A* - A-Star Search
#
# **Tu tarea:** Completa la implementación de A*.
#
# Recuerda:
# - `f(n) = g(n) + h(n)` donde:
#   - `g(n)` = costo acumulado desde start
#   - `h(n)` = heurística (distancia Manhattan al goal)
# - Usa una **cola de prioridad** (`heapq`) ordenada por `f`
# - La heurística Manhattan ya está implementada


# %%
def manhattan(a: Position, b: Position) -> int:
    """Distancia Manhattan entre dos posiciones."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(
    maze: List[List[int]], start: Position, goal: Position
) -> Tuple[Optional[List[Position]], int]:
    """
    A* Search con heurística Manhattan.

    Args:
        maze: Matriz del laberinto
        start: Posición inicial
        goal: Posición objetivo

    Returns:
        Tupla (camino, nodos_expandidos)
    """
    # TODO: Implementa A*

    # Cola de prioridad: (f_score, g_score, posición)
    # heapq ordena por el primer elemento (f_score)
    open_set = [(manhattan(start, goal), 0, start)]

    # Diccionario de g_scores (costo acumulado)
    g_score: Dict[Position, int] = {start: 0}

    # Para reconstruir el camino
    came_from: Dict[Position, Optional[Position]] = {start: None}

    expanded = 0

    # Pista conceptual:
    # g(n): costo acumulado
    # h(n): manhattan
    # f(n): g(n) + h(n)

    while open_set:
        # TODO: Sacar el nodo con menor f de la cola
        f, g, current = heapq.heappop(open_set)
        expanded += 1

        # TODO: Verificar objetivo

        # TODO: Expandir vecinos
        # Para cada vecino:
        #   - Calcular tentative_g = g + 1
        #   - Si es mejor que el g_score guardado:
        #     - Actualizar g_score
        #     - Calcular f = tentative_g + h(neighbor, goal)
        #     - Agregar a open_set
        #     - Guardar came_from

        pass  # <-- ELIMINAR

    return None, expanded


# %% [markdown]
# ## 6. Función para reconstruir el camino
#
# Esta función ya está completa. Úsala en tus implementaciones.


# %%
def reconstruct_path(
    parent: Dict[Position, Optional[Position]], goal: Position
) -> List[Position]:
    """
    Reconstruye el camino desde el goal hasta el start usando el diccionario parent.

    Args:
        parent: Diccionario {posición: posición_padre}
        goal: Posición final

    Returns:
        Lista de posiciones desde start hasta goal
    """
    path = []
    current: Optional[Position] = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    return path[::-1]  # Invertir para ir de start a goal


# %% [markdown]
# ## 7. Verificación del camino


# %%
def verify_path(
    maze: List[List[int]], path: List[Position], start: Position, goal: Position
) -> Tuple[bool, str]:
    """Verifica que un camino sea válido."""
    if not path:
        return False, "Camino vacío"

    if path[0] != start:
        return False, f"El camino no empieza en start {start}"

    if path[-1] != goal:
        return False, f"El camino no termina en goal {goal}"

    rows, cols = len(maze), len(maze[0])

    for i, (r, c) in enumerate(path):
        if not (0 <= r < rows and 0 <= c < cols):
            return False, f"Posición {(r, c)} fuera de límites"

        if maze[r][c] == 1:
            return False, f"Posición {(r, c)} es una pared"

        if i > 0:
            pr, pc = path[i - 1]
            dr, dc = abs(r - pr), abs(c - pc)
            if not ((dr == 1 and dc == 0) or (dr == 0 and dc == 1)):
                return False, f"Movimiento inválido de {(pr, pc)} a {(r, c)}"

    return True, f"✓ Camino válido de longitud {len(path)}"


# %% [markdown]
# ## 8. Pruebas y Comparación
#
# Ejecuta esta celda después de implementar los algoritmos.


# %%
def run_comparison(maze: List[List[int]], name: str):
    """Ejecuta los tres algoritmos y compara resultados."""
    n = len(maze)
    start = (0, 0)
    goal = (n - 1, n - 1)

    print(f"\n{'=' * 50}")
    print(f"LABERINTO {name}")
    print(f"{'=' * 50}")
    print(f"Tamaño: {n}x{n}")
    print(f"Start: {start}, Goal: {goal}")
    print()

    results = {}

    # BFS
    print("Ejecutando BFS...")
    path_bfs, exp_bfs = bfs(maze, start, goal)
    if path_bfs:
        valid, msg = verify_path(maze, path_bfs, start, goal)
        results["BFS"] = {
            "path_length": len(path_bfs),
            "expanded": exp_bfs,
            "valid": valid,
        }
        print(f"  Longitud del camino: {len(path_bfs)}")
        print(f"  Nodos expandidos: {exp_bfs}")
        print(f"  Validación: {msg}")
    else:
        print("  No se encontró camino")

    # DFS
    print("\nEjecutando DFS...")
    path_dfs, exp_dfs = dfs(maze, start, goal)
    if path_dfs:
        valid, msg = verify_path(maze, path_dfs, start, goal)
        results["DFS"] = {
            "path_length": len(path_dfs),
            "expanded": exp_dfs,
            "valid": valid,
        }
        print(f"  Longitud del camino: {len(path_dfs)}")
        print(f"  Nodos expandidos: {exp_dfs}")
        print(f"  Validación: {msg}")
    else:
        print("  No se encontró camino")

    # A*
    print("\nEjecutando A*...")
    path_astar, exp_astar = astar(maze, start, goal)
    if path_astar:
        valid, msg = verify_path(maze, path_astar, start, goal)
        results["A*"] = {
            "path_length": len(path_astar),
            "expanded": exp_astar,
            "valid": valid,
        }
        print(f"  Longitud del camino: {len(path_astar)}")
        print(f"  Nodos expandidos: {exp_astar}")
        print(f"  Validación: {msg}")
    else:
        print("  No se encontró camino")

    # Resumen
    print("\n" + "-" * 50)
    print("RESUMEN:")
    print("-" * 50)
    print(f"{'Algoritmo':<12} {'Camino':<12} {'Expandidos':<12} {'Óptimo?'}")
    print("-" * 50)

    optimal_length = results.get("BFS", {}).get("path_length", float("inf"))

    for algo in ["BFS", "DFS", "A*"]:
        if algo in results:
            r = results[algo]
            is_optimal = "✓" if r["path_length"] == optimal_length else "✗"
            print(f"{algo:<12} {r['path_length']:<12} {r['expanded']:<12} {is_optimal}")

    return results


# Ejecutar comparación (descomenta cuando tengas las implementaciones)

# Caso simple (para debug)
# results_10 = run_comparison(MAZE_10, "10x10")

# Caso intermedio
# results_15 = run_comparison(MAZE_15, "15x15")

# Caso más desafiante
# results_20 = run_comparison(MAZE_20, "20x20")

# Extra opcional (no requerido)
# results_25 = run_comparison(MAZE_25_EXTRA, "25x25 (extra)")

# %% [markdown]
# ## 9. Visualización (Opcional)
#
# Si quieres ver el camino visualmente:


# %%
def visualize_path(
    maze: List[List[int]], path: Optional[List[Position]], title: str = ""
):
    """Visualiza el laberinto con el camino encontrado."""
    n = len(maze)
    path_set = set(path) if path else set()

    print(f"\n{title}")
    print("+" + "-" * (n * 2 - 1) + "+")

    for r in range(n):
        row_str = "|"
        for c in range(n):
            if (r, c) == (0, 0):
                row_str += "S"
            elif (r, c) == (n - 1, n - 1):
                row_str += "G"
            elif (r, c) in path_set:
                row_str += "·"
            elif maze[r][c] == 1:
                row_str += "█"
            else:
                row_str += " "
            if c < n - 1:
                row_str += " "
        row_str += "|"
        print(row_str)

    print("+" + "-" * (n * 2 - 1) + "+")

    if path:
        print(f"Longitud del camino: {len(path)}")


# Ejemplo de uso (descomenta cuando tengas implementación):
# path_bfs, _ = bfs(MAZE_15, (0,0), (14,14))
# visualize_path(MAZE_15, path_bfs, "BFS en MAZE_15")

# %% [markdown]
# ## 10. Preguntas para el Informe
#
# Responde estas preguntas en tu informe (½-1 página):
#
# 1. **¿Por qué A* expande menos nodos que BFS si ambos encuentran el camino óptimo?**
#
# 2. **¿Qué observaste con DFS?** ¿Encontró el camino óptimo? ¿Cuántos nodos expandió comparado con BFS?
#
# 3. **¿Qué pasaría si la heurística NO fuera admisible?** (Ej: h(n) = 2 * Manhattan)
#
# 4. **¿En qué casos prácticos preferirías DFS sobre BFS?**
