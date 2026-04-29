def reconstruct_path(parent, goal):
    path = []

    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    return path[::-1]
