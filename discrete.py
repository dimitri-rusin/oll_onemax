import gymnasium
import numpy as np
import os
import plotly.graph_objects as go
import sqlite3

if __name__ == '__main__':
  global_seed = 42
  env_seed = 100
  agent_seed = 200
  n = 25
  learning_rate = 0.1
  gamma = 0.9
  epsilon = 0.1
  episodes = 80
  evaluation_interval = 10  # Interval to evaluate policy

  # Set the global random seed
  np.random.seed(global_seed)

class OneMaxEnv(gymnasium.Env):
    def __init__(self, n, seed=None):
        super(OneMaxEnv, self).__init__()
        self.n = n
        self.action_space = gymnasium.spaces.Discrete(n)
        self.observation_space = gymnasium.spaces.Box(low=0, high=n - 1, shape=(1,), dtype=np.int32)
        self.seed = seed
        self.random = np.random.RandomState(self.seed)
        assert seed is None
        self.optimum = None
        self.current_solution = None
        self.evaluations = {}

    def reset(self, episode_seed):
        # Use the provided seed for reproducibility
        self.seed = episode_seed
        self.random = np.random.RandomState(self.seed)

        self.current_solution = np.zeros(self.n, dtype=int)
        self.optimum = self.random.randint(2, size=self.n)

        # Set approximately 85% of the bits to 1
        num_ones = int(self.n * 0.5)
        one_positions = self.random.choice(self.n, num_ones, replace=False)
        self.current_solution[one_positions] = 1

        self.evaluations = {}
        return np.array([self.evaluate(self.current_solution)])

    def step(self, action):
        λ = action + 1
        offspring = self.generate_offspring(λ)
        self.current_solution, evaluations_this_step = self.select_solution(offspring)
        fitness = self.evaluate(self.current_solution)
        reward = -evaluations_this_step
        done = fitness == self.n
        return np.array([fitness]), reward, done, {}

    def evaluate(self, solution):
        solution_key = tuple(solution)
        if solution_key in self.evaluations:
            return self.evaluations[solution_key]
        fitness = np.sum(solution)
        self.evaluations[solution_key] = fitness
        return fitness

    def generate_offspring(self, λ):
        offspring = []
        for _ in range(λ):
            mutated = self.mutate(self.current_solution)
            crossed = self.crossover(self.current_solution, mutated)
            offspring.append(crossed)
        return offspring

    def mutate(self, solution):
        mutation = self.random.randint(2, size=self.n)
        return np.bitwise_xor(solution, mutation)

    def crossover(self, parent, other):
        mask = self.random.randint(2, size=self.n)
        return np.where(mask, parent, other)

    def select_solution(self, offspring):
        evaluations_this_step = 0
        best_solution = self.current_solution
        best_fitness = self.evaluate(self.current_solution)
        for child in offspring:
            fitness = self.evaluate(child)
            if fitness > best_fitness:
                best_solution = child
                best_fitness = fitness
            evaluations_this_step += 1
        return best_solution, evaluations_this_step



def init_q_table(n):
  return np.zeros((n, n))

def choose_action(state, q_table, epsilon, n, random_state):
  if random_state.random() < epsilon:
    return random_state.randint(n)
  else:
    return np.argmax(q_table[state])

def q_learning_and_save_policy(env, total_episodes, learning_rate, gamma, epsilon, seed, conn):
    random_state = np.random.RandomState(seed)
    q_table = init_q_table(env.n)

    for episode in range(1, total_episodes + 1):
        print("training episode", episode)
        episode_seed = random_state.randint(100000)
        state = env.reset(episode_seed)[0]
        done = False

        while not done:
            action = choose_action(state, q_table, epsilon, env.n, random_state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[0]

            q_predict = q_table[state, action]
            q_target = reward if done else reward + gamma * np.max(q_table[next_state])
            q_table[state, action] += learning_rate * (q_target - q_predict)
            state = next_state

        # Save the policy induced by the Q-table at regular intervals
        if episode % evaluation_interval == 0:
            policy = np.argmax(q_table, axis=1)
            policy_id = insert_policy_and_get_id(conn, policy)
            insert_policy_info(conn, policy_id, episode)

    return q_table

# Function to visualize the Q-table using Plotly with highlighted exploiting actions
def plot_q_table(q_table):
  heatmap = go.Heatmap(
    z=q_table,
    x=[f'λ={i+1}' for i in range(q_table.shape[1])],
    y=[f'State={i}' for i in range(q_table.shape[0])],
    hoverongaps=False
  )
  exploiting_actions = np.argmax(q_table, axis=1)
  markers_x = [f'λ={action+1}' for action in exploiting_actions]
  markers_y = [f'State={i}' for i in range(len(exploiting_actions))]
  markers = go.Scatter(
    x=markers_x, y=markers_y, mode='markers',
    marker=dict(color='black', size=10, line=dict(color='white', width=2)),
    showlegend=False
  )
  fig = go.Figure(data=[heatmap, markers])
  fig.update_layout(title='Q-Table Heatmap with Highlighted Exploiting Actions',
                    xaxis_title='Actions', yaxis_title='States')
  fig.show()

# Create tables with auto-increment policy_id and include seed in episode_evaluation
def create_tables(conn):
  with conn:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS policy_log (
            policy_id INTEGER,
            fitness INTEGER,
            lambda INTEGER
        );
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS episode_evaluation (
            policy_id INTEGER,
            steps INTEGER,
            episode_seed INTEGER,
            FOREIGN KEY(policy_id) REFERENCES policy_log(policy_id)
        );
    ''')
    conn.execute('''
      CREATE TABLE IF NOT EXISTS policy_info (
        policy_id INTEGER PRIMARY KEY AUTOINCREMENT,
        num_training_episodes INTEGER,
        FOREIGN KEY(policy_id) REFERENCES policy_log(policy_id)
      );
    ''')

def insert_policy_and_get_id(conn, policy):
  cursor = conn.cursor()
  cursor.execute('INSERT INTO policy_info (num_training_episodes) VALUES (?);', (0,))
  policy_id = cursor.lastrowid
  for fitness, action in enumerate(policy):
    cursor.execute('INSERT INTO policy_log (policy_id, fitness, lambda) VALUES (?, ?, ?);',
                   (policy_id, fitness, int(action + 1)))
  conn.commit()
  return policy_id

def insert_policy_info(conn, policy_id, num_training_episodes):
  with conn:
    cursor = conn.cursor()
    cursor.execute('UPDATE policy_info SET num_training_episodes = ? WHERE policy_id = ?;',
                   (num_training_episodes, policy_id))

def drop_all_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve a list of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")
    tables = cursor.fetchall()

    # Drop each table
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")

    # Reset the sqlite_sequence table
    cursor.execute("DELETE FROM sqlite_sequence;")

    conn.commit()
    conn.close()

if __name__ == '__main__':
    db_path = 'ppo_data/policy_data.db'
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    create_tables(conn)

    agent_seed = np.random.randint(0, 100000)
    env = OneMaxEnv(n=n)

    q_table = q_learning_and_save_policy(env, episodes, learning_rate, gamma, epsilon, agent_seed, conn)

    plot_q_table(q_table)
    conn.close()
