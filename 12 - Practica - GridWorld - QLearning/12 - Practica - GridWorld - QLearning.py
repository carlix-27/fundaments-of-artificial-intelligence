
"""
TP Bloque 12 - Q-Learning en GridWorld
Fundamentos e Historia de la IA

Codigo completo para ejecutar, observar e interpretar.
No hace falta reescribir el algoritmo: la tarea consiste en identificar las partes del ciclo,
modificar hiperparametros e interpretar resultados.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List

State = Tuple[int, int]

ACTIONS = {
    0: (-1, 0),  # arriba
    1: (1, 0),   # abajo
    2: (0, -1),  # izquierda
    3: (0, 1),   # derecha
}
ACTION_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

@dataclass
class GridWorld:
    rows: int = 5
    cols: int = 5
    start: State = (0, 0)
    goal: State = (4, 4)
    trap: State = (3, 4)
    walls: Tuple[State, ...] = ((2, 2),)
    step_reward: float = -1.0
    goal_reward: float = 10.0
    trap_reward: float = -10.0
    max_steps: int = 100

    def reset(self) -> State:
        return self.start

    def state_to_index(self, state: State) -> int:
        return state[0] * self.cols + state[1]

    def index_to_state(self, idx: int) -> State:
        return divmod(idx, self.cols)

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    def in_bounds(self, state: State) -> bool:
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, state: State) -> bool:
        return state in self.walls

    def is_terminal(self, state: State) -> bool:
        return state == self.goal or state == self.trap

    def step(self, state: State, action: int):
        """Ejecuta una accion y devuelve: nuevo_estado, recompensa, terminado."""
        if self.is_terminal(state):
            return state, 0.0, True

        dr, dc = ACTIONS[action]
        candidate = (state[0] + dr, state[1] + dc)

        # Si choca contra pared o borde, se queda en el mismo estado.
        if not self.in_bounds(candidate) or self.is_wall(candidate):
            candidate = state

        if candidate == self.goal:
            return candidate, self.goal_reward, True
        if candidate == self.trap:
            return candidate, self.trap_reward, True
        return candidate, self.step_reward, False


def choose_action_epsilon_greedy(Q: np.ndarray, state_idx: int, epsilon: float, rng: np.random.Generator) -> int:
    """
    ESTRATEGIA DE ACCION: epsilon-greedy.
    - Con probabilidad epsilon: explora una accion aleatoria.
    - Con probabilidad 1-epsilon: explota la mejor accion conocida.
    """
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[state_idx]))


def train_q_learning(env: GridWorld, alpha=0.1, gamma=0.9, epsilon=0.1,
                     episodes=1000, seed=7, epsilon_decay=None):
    """
    Entrena un agente Q-Learning.

    CICLO:
    1. estado s
    2. epsilon-greedy elige accion a
    3. entorno devuelve recompensa r y nuevo estado s'
    4. Q-Learning actualiza Q(s,a)
    5. repetir
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode: List[float] = []
    steps_per_episode: List[int] = []

    eps = epsilon
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(env.max_steps):
            s_idx = env.state_to_index(state)

            # 1) ELEGIR ACCION: epsilon-greedy
            action = choose_action_epsilon_greedy(Q, s_idx, eps, rng)

            # 2) EJECUTAR ACCION: el entorno responde
            next_state, reward, done = env.step(state, action)
            ns_idx = env.state_to_index(next_state)

            # 3) ACTUALIZAR Q: target bellmaniano
            target = reward + gamma * np.max(Q[ns_idx]) * (not done)
            error = target - Q[s_idx, action]
            Q[s_idx, action] = Q[s_idx, action] + alpha * error

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        if epsilon_decay is not None:
            eps = max(0.01, eps * epsilon_decay)

    return Q, np.array(rewards_per_episode), np.array(steps_per_episode)


def extract_policy(Q: np.ndarray, env: GridWorld) -> np.ndarray:
    policy = np.empty((env.rows, env.cols), dtype=object)
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state == env.goal:
                policy[r, c] = "META"
            elif state == env.trap:
                policy[r, c] = "TRAMPA"
            elif state in env.walls:
                policy[r, c] = "PARED"
            else:
                idx = env.state_to_index(state)
                policy[r, c] = ACTION_SYMBOLS[int(np.argmax(Q[idx]))]
    return policy


def print_policy(policy: np.ndarray):
    for row in policy:
        print("  ".join(f"{x:>6}" for x in row))


def plot_rewards(rewards: np.ndarray, title="Recompensa por episodio", window=50):
    plt.figure(figsize=(9, 4))
    plt.plot(rewards, alpha=0.35, label="recompensa")
    if len(rewards) >= window:
        moving = np.convolve(rewards, np.ones(window)/window, mode="valid")
        plt.plot(np.arange(window-1, len(rewards)), moving, label=f"promedio movil {window}")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa total")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_policy(policy: np.ndarray, env: GridWorld, title="Politica aprendida"):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_xticks(np.arange(env.cols+1))
    ax.set_yticks(np.arange(env.rows+1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for r in range(env.rows):
        for c in range(env.cols):
            txt = policy[r, c]
            ax.text(c + 0.5, r + 0.5, txt, ha="center", va="center", fontsize=13)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def run_experiment(name: str, alpha: float, gamma: float, epsilon: float, episodes=1000, seed=7, epsilon_decay=None):
    env = GridWorld()
    Q, rewards, steps = train_q_learning(env, alpha=alpha, gamma=gamma, epsilon=epsilon,
                                         episodes=episodes, seed=seed, epsilon_decay=epsilon_decay)
    policy = extract_policy(Q, env)
    print(f"\n=== {name} ===")
    print(f"alpha={alpha}, gamma={gamma}, epsilon={epsilon}, episodes={episodes}, epsilon_decay={epsilon_decay}")
    print(f"Recompensa promedio ultimos 100 episodios: {np.mean(rewards[-100:]):.2f}")
    print(f"Pasos promedio ultimos 100 episodios: {np.mean(steps[-100:]):.2f}")
    print_policy(policy)
    plot_rewards(rewards, title=f"{name} - Recompensa por episodio")
    plot_policy(policy, env, title=f"{name} - Politica aprendida")
    return {"name": name, "Q": Q, "rewards": rewards, "steps": steps, "policy": policy}


if __name__ == "__main__":
    # Configuracion base
    base = run_experiment("Base", alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000)

    # Experimentos sugeridos. Activar/desactivar segun necesidad.
    experiments = [
        ("Sin exploracion", 0.1, 0.9, 0.0),
        ("Exploracion alta", 0.1, 0.9, 0.8),
        ("Futuro debil", 0.1, 0.2, 0.1),
        ("Aprendizaje agresivo", 0.9, 0.9, 0.1),
    ]
    for name, alpha, gamma, epsilon in experiments:
        run_experiment(name, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=1000)
