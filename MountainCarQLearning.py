import numpy as np
import gym
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
n_actions = env.action_space.n   # 3 acciones

Q_table = np.zeros(n_states + (n_actions,))

# ============================================================
# Parámetros de entrenamiento
# ============================================================
alpha = 0.1         # learning rate
gamma = 0.99        # descuento
epsilon = 1.0       # exploración inicial
epsilon_min = 0.05
epsilon_decay = 0.995
episodes = 5000

reward_history = []  # Para graficar en Flask


# ============================================================
# ENTRENAMIENTO Q-LEARNING
# ============================================================
def train_mountaincar():
    global Q_table, epsilon

    for episode in range(episodes):
        state = discretize_state(env.reset()[0])
        total_reward = 0

        done = False
        while not done:
            # Política ε-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])

            next_state_raw, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state_raw)

            # Q-learning update
            best_next = np.max(Q_table[next_state])
            Q_table[state + (action,)] += alpha * (reward + gamma * best_next - Q_table[state + (action,)])

            state = next_state
            total_reward += reward

        reward_history.append(total_reward)

        # Reducir ε
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    # Guardar modelo
    with open("static/mountaincar_qtable.pkl", "wb") as f:
        pickle.dump(Q_table, f)

    return reward_history


# ============================================================
# Función para cargar modelo
# ============================================================
def load_model():
    global Q_table
    if os.path.exists("static/mountaincar_qtable.pkl"):
        with open("static/mountaincar_qtable.pkl", "rb") as f:
            Q_table = pickle.load(f)
        return True
    return False


# ============================================================
# Probar política aprendida (para Flask)
# ============================================================
def run_policy(max_steps=500):
    """Devuelve la trayectoria observada del agente."""
    load_model()

    state = discretize_state(env.reset()[0])
    trajectory = []

    for _ in range(max_steps):
        action = np.argmax(Q_table[state])
        next_state_raw, reward, done, _, _ = env.step(action)
        trajectory.append([next_state_raw[0], next_state_raw[1]])  # pos, vel

        state = discretize_state(next_state_raw)
        if done:
            break

    return trajectory
