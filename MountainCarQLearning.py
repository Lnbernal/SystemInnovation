import numpy as np
import gymnasium as gym
import pickle
import os

# ============================================================
# Configuración del entorno Mountain Car
# ============================================================
env = gym.make("MountainCar-v0")

# Rango de estados continuo → lo discretizamos
pos_space = np.linspace(env.observation_space.low[0],
                        env.observation_space.high[0], 40)
vel_space = np.linspace(env.observation_space.low[1],
                        env.observation_space.high[1], 40)

def discretize_state(state):
    """Convierte un estado continuo en índices discretos."""
    pos, vel = state
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)
    return (pos_bin, vel_bin)


# ============================================================
# Inicializar Q-Table
# ============================================================
n_states = (len(pos_space) + 1, len(vel_space) + 1)
n_actions = env.action_space.n

Q_table = np.zeros(n_states + (n_actions,))


# ============================================================
# Parámetros de entrenamiento
# ============================================================
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
episodes = 3000

reward_history = []


# ============================================================
# ENTRENAMIENTO Q-LEARNING
# ============================================================
def train_mountaincar():
    global Q_table, epsilon

    for episode in range(episodes):

        obs, info = env.reset()
        state = discretize_state(obs)

        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):

            # Política e-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_obs)

            best_next = np.max(Q_table[next_state])
            Q_table[state + (action,)] += alpha * (
                reward + gamma * best_next - Q_table[state + (action,)]
            )

            state = next_state
            total_reward += reward

        reward_history.append(total_reward)

        # Reducir exploración
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    # Guardar Q-table
    with open("static/mountaincar_qtable.pkl", "wb") as f:
        pickle.dump(Q_table, f)

    return reward_history


# ============================================================
# Cargar modelo
# ============================================================
def load_model():
    global Q_table
    if os.path.exists("static/mountaincar_qtable.pkl"):
        with open("static/mountaincar_qtable.pkl", "rb") as f:
            Q_table = pickle.load(f)
        return True
    return False


# ============================================================
# Probar política aprendida
# ============================================================
def run_policy(max_steps=500):
    load_model()

    obs, info = env.reset()
    state = discretize_state(obs)

    trajectory = []
    action_count = {0: 0, 1: 0, 2: 0}

    for _ in range(max_steps):
        action = np.argmax(Q_table[state])
        action_count[action] += 1

        next_obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append([next_obs[0], next_obs[1]])

        state = discretize_state(next_obs)

        if terminated or truncated:
            break

    return trajectory, action_count