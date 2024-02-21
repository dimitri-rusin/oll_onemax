import gymnasium
import multiprocessing
import numpy
import numpy as np
import os
import sqlite3
import sys
import tuning_environments
import yaml

config = None



def init_q_table(n):
  """Initialize a Q-table with zeros."""
  return numpy.zeros((n, n))

def choose_action(state, q_table, epsilon, n, random_state):
  """Choose an action based on the current state and Q-table."""
  return random_state.randint(n) if random_state.random() < epsilon else numpy.argmax(q_table[state])

def update_q_table(state, action, reward, next_state, done, q_table, learning_rate, gamma):
  """Update the Q-table using the Q-learning algorithm."""
  q_predict = q_table[state, action]
  q_target = reward if done else reward + gamma * numpy.max(q_table[next_state])
  q_table[state, action] += learning_rate * (q_target - q_predict)

def q_learning_and_save_policy(env, total_episodes, learning_rate, gamma, epsilon, seed, conn, evaluation_interval):
  """Perform Q-learning and save the policy to the database."""
  random_state = numpy.random.RandomState(seed)
  q_table = init_q_table(env.n)

  for episode in range(1, total_episodes + 1):
    print("Training episode", episode)
    episode_seed = random_state.randint(100_000)
    state, done = env.reset(episode_seed)[0], False

    while not done:
      action = choose_action(state, q_table, epsilon, env.n, random_state)
      next_state, reward, done, _ = env.step(action)
      update_q_table(state, action, reward, next_state[0], done, q_table, learning_rate, gamma)
      state = next_state[0]

    if episode % evaluation_interval == 0:
      save_policy(conn, q_table, episode)

  return q_table

def create_tables(conn):
  """Create necessary tables in the database."""
  with conn:
    conn.executescript('''
      CREATE TABLE IF NOT EXISTS policy_log (policy_id INTEGER, fitness INTEGER, lambda INTEGER);
      CREATE TABLE IF NOT EXISTS episode_evaluation (policy_id INTEGER, steps INTEGER, episode_seed INTEGER, FOREIGN KEY(policy_id) REFERENCES policy_log(policy_id));
      CREATE TABLE IF NOT EXISTS policy_info (policy_id INTEGER PRIMARY KEY AUTOINCREMENT, num_training_episodes INTEGER, FOREIGN KEY(policy_id) REFERENCES policy_log(policy_id));
    ''')

def save_policy(conn, q_table, episode):
  """Save the policy induced by the Q-table and launch a process for evaluation."""
  policy_id = insert_policy_and_get_id(conn, numpy.argmax(q_table, axis=1))
  insert_policy_info(conn, policy_id, episode)
  process = multiprocessing.Process(target=evaluate_policy, args=(policy_id, config['db_path'], config['n'], config['env_seed']))
  process.start()

def insert_policy_and_get_id(conn, policy):
  """Insert policy into policy_log and return the generated policy_id."""
  cursor = conn.cursor()
  cursor.execute('INSERT INTO policy_info (num_training_episodes) VALUES (?);', (0,))
  policy_id = cursor.lastrowid
  cursor.executemany('INSERT INTO policy_log (policy_id, fitness, lambda) VALUES (?, ?, ?);', [(policy_id, fitness, int(action + 1)) for fitness, action in enumerate(policy)])
  conn.commit()
  return policy_id



def evaluate_policy(policy_id, db_path, n, env_seed):
    """Evaluate the policy from the database over multiple episodes."""
    conn = sqlite3.connect(db_path)
    policy = fetch_policy(conn, policy_id)
    conn.close()

    # Print the policy
    print(f"Policy {policy_id} assignment: {policy}")

    fitness_results = []
    max_step_index = 0

    random_state = numpy.random.RandomState(env_seed)
    # Evaluate the policy over 10 episodes
    env = tuning_environments.OneMaxEnv(n=n)
    for episode in range(10):
        episode_seed = random_state.randint(100_000)
        fitness, max_steps = evaluate_episode(env, policy, episode_seed)
        fitness_results.append(fitness)
        max_step_index = max(max_step_index, max_steps)

    # Print fitness values for each step across all episodes
    for step in range(1, max_step_index + 1):
        step_fitnesses = [episode_fitness[step - 1] if step <= len(episode_fitness) else 'N/A' for episode_fitness in fitness_results]
        print(f"Step {step} fitnesses across episodes: {step_fitnesses}")

def evaluate_episode(env, policy, episode_seed):
    """Simulate an episode based on the policy and return fitness at each step."""
    state = env.reset(episode_seed)[0]
    done = False
    fitness_values = []

    while not done:
        action = policy[state]
        next_state, _, done, _ = env.step(action)
        state = next_state[0]
        fitness = env.evaluate(env.current_solution)
        fitness_values.append(fitness)

    return fitness_values, len(fitness_values)



def fetch_policy(conn, policy_id):
    """Fetch a policy from the database and reconstruct it as a dictionary."""
    cursor = conn.cursor()
    cursor.execute('SELECT fitness, lambda FROM policy_log WHERE policy_id = ?', (policy_id,))
    rows = cursor.fetchall()
    return {fitness: lambda_val for fitness, lambda_val in rows}



def insert_policy_info(conn, policy_id, num_training_episodes):
  """Update policy_info with the number of training episodes."""
  with conn:
    conn.execute('UPDATE policy_info SET num_training_episodes = ? WHERE policy_id = ?;', (num_training_episodes, policy_id))

def drop_all_tables(db_path):
  """Drop all tables from the database."""
  with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    cursor.executescript("DROP TABLE IF EXISTS policy_log; DROP TABLE IF EXISTS episode_evaluation; DROP TABLE IF EXISTS policy_info;")
    cursor.execute("DELETE FROM sqlite_sequence;")

def main():
  try:
    with open(".env.yaml") as file:
      global config
      config = yaml.safe_load(file)
  except FileNotFoundError:
    print("Error: '.env.yaml' does not exist.", file=sys.stderr)
    sys.exit(1)

  setup_config(config)
  conn = setup_database(config['db_path'])
  create_tables(conn)

  env = tuning_environments.OneMaxEnv(n=config['n'])
  q_table = q_learning_and_save_policy(env, config['episodes'], config['learning_rate'], config['gamma'], config['epsilon'], numpy.random.randint(0, 100_000), conn, config['evaluation_interval'])

  conn.close()

def setup_config(config):
  """Setup global configuration parameters."""
  numpy.random.seed(config['global_seed'])

def setup_database(db_path):
  """Prepare the database, creating necessary directories and tables."""
  drop_all_tables(db_path)

  directory_path = os.path.dirname(db_path)
  if not os.path.exists(directory_path):
    os.makedirs(directory_path)

  return sqlite3.connect(db_path)

if __name__ == '__main__':
  main()
