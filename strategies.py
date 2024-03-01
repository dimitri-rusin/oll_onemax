import gymnasium
import numpy
import os
import paper_code.onell_algs
import sqlite3
import sys
import time
import yaml

config = None





# ============== ENVIRONMENT - BEGIN ==============

class OneMaxOLL(gymnasium.Env):
  def __init__(self, n, seed=None):
    super(OneMaxOLL, self).__init__()
    self.n = n
    self.action_space = gymnasium.spaces.Discrete(n)
    self.observation_space = gymnasium.spaces.Box(low=0, high=n - 1, shape=(1,), dtype=numpy.int32)
    self.seed = seed
    self.random = numpy.random.RandomState(self.seed)
    self.random_number_generator = numpy.random.default_rng(self.seed)
    assert seed is None
    self.current_solution = None

  def reset(self, episode_seed):
    # Use the provided seed for reproducibility
    self.seed = episode_seed
    self.num_function_evaluations = 0
    self.random = numpy.random.RandomState(self.seed)
    self.random_number_generator = numpy.random.default_rng(self.seed)
    self.current_solution = paper_code.onell_algs.OneMax(self.n, rng = self.random_number_generator)

    # Set approximately 85% of the bits to 1
    num_ones = int(self.n * 0.5)
    one_positions = self.random.choice(self.n, num_ones, replace=False)
    self.current_solution.data[one_positions] = True
    self.current_solution.eval()

    return numpy.array([self.current_solution.fitness])

  def step(self, λ):
    if isinstance(λ, numpy.ndarray) and λ.size == 1:
      λ = λ.item()
    λ += 1
    p = λ / self.n
    population_size = numpy.round(λ).astype(int)
    prior_fitness = self.current_solution.fitness
    xprime, f_xprime, ne1 = self.current_solution.mutate(p, population_size, self.random_number_generator)

    c = 1 / λ
    y, f_y, ne2 = self.current_solution.crossover(
      xprime,
      c,
      population_size,
      True,
      True,
      self.random_number_generator,
    )

    if f_y >= self.current_solution.fitness:
      self.current_solution = y

    num_evaluations_of_this_step = int(ne1 + ne2)
    reward = -num_evaluations_of_this_step
    terminated = self.current_solution.is_optimal()
    info = {}

    self.num_function_evaluations += num_evaluations_of_this_step

    return numpy.array([self.current_solution.fitness]), reward, terminated, info

# ============== ENVIRONMENT - END ==============



def q_learning_and_save_policy(env, total_episodes, learning_rate, gamma, epsilon, seed, conn, evaluation_interval):
  """Perform Q-learning, update Q-table, choose actions, save and evaluate policies."""

  # Initialize a Q-table with zeros
  q_table = numpy.zeros((env.n, env.n))
  num_q_table_updates = 0

  # Setup random state
  random_state = numpy.random.RandomState(seed)

  # Save the policy induced by the initial Q-table and launch a process for evaluation
  policy = [max(1, int(numpy.sqrt(action))) for action in numpy.argmax(q_table, axis=1)]
  policy_id = insert_policy_and_get_id(conn, policy)
  insert_policy_info(conn, policy_id, 0, num_q_table_updates)
  evaluate_policy(
    policy_id,
    config['db_path'],
    config['n'],
    config['env_seed'],
    config['num_evaluation_episodes'],
  )

  for episode in range(1, total_episodes + 1):
    print("Training episode", episode)
    episode_seed = random_state.randint(100_000)
    state, done = env.reset(episode_seed)[0], False

    while not done:
      # Choose an action based on the current state and Q-table
      action = random_state.randint(env.n) if random_state.random() < epsilon else numpy.argmax(q_table[state])

      # Perform the action
      next_state, reward, done, _ = env.step(action)

      # Update the Q-table using the Q-learning algorithm
      q_predict = q_table[state, action]
      q_target = reward if done else reward + gamma * numpy.max(q_table[next_state[0]])
      q_table[state, action] += learning_rate * (q_target - q_predict)
      num_q_table_updates += 1

      state = next_state[0]

      if episode % evaluation_interval == 0:
          policy = [max(1, int(numpy.sqrt(action))) for action in numpy.argmax(q_table, axis=1)]
          policy_id = insert_policy_and_get_id(conn, policy)
          insert_policy_info(conn, policy_id, episode, num_q_table_updates)
          evaluate_policy(policy_id, config['db_path'], config['n'], config['env_seed'], config['num_evaluation_episodes'])

  return q_table

def create_tables(conn):
  """Create necessary tables in the database."""
  with conn:
    conn.executescript('''
      CREATE TABLE IF NOT EXISTS policies_data (policy_id INTEGER, fitness INTEGER, lambda INTEGER);
      CREATE TABLE IF NOT EXISTS policies_info (policy_id INTEGER PRIMARY KEY AUTOINCREMENT, num_training_episodes INTEGER, num_q_table_updates INTEGER, FOREIGN KEY(policy_id) REFERENCES policies_data(policy_id));
      CREATE TABLE IF NOT EXISTS episode_info (policy_id INTEGER, episode_id INTEGER PRIMARY KEY AUTOINCREMENT, episode_seed INTEGER, episode_length INTEGER, num_function_evaluations INTEGER, FOREIGN KEY(policy_id) REFERENCES policies_data(policy_id));
    ''')

def insert_special_policy(conn, num_dimensions):
  """Insert the special policy with policy_id -1."""
  policy_lambdas = [int(numpy.sqrt(num_dimensions / (num_dimensions - fitness))) for fitness in range(num_dimensions)]
  policy_id = -1  # Special policy ID
  insert_policy_and_get_id(conn, policy_lambdas, policy_id)

def insert_policy_and_get_id(conn, policy, policy_id=None):
  """Insert policy into policies_data and return the generated policy_id."""
  retry_count = 0
  max_retries = 5
  while True:
    try:
      with conn:  # This automatically begins and commits/rollbacks a transaction
        cursor = conn.cursor()
        if policy_id is None:
          cursor.execute('INSERT INTO policies_info (num_training_episodes) VALUES (?);', (0,))
          policy_id = cursor.lastrowid
        else:
          cursor.execute('INSERT INTO policies_info (policy_id, num_training_episodes) VALUES (?, ?);', (policy_id, 0,))
        cursor.executemany('INSERT INTO policies_data (policy_id, fitness, lambda) VALUES (?, ?, ?);',
                           [(policy_id, fitness, action) for fitness, action in enumerate(policy)])
      break
    except sqlite3.OperationalError as e:
      if retry_count < max_retries:
        retry_count += 1
        time.sleep(1)  # Wait for 1 second before retrying
      else:
        raise e
  return policy_id

def evaluate_policy(policy_id, db_path, n, env_seed, num_evaluation_episodes):
  # Open a new connection for each process
  with sqlite3.connect(db_path, timeout=10) as conn:  # Timeout set to 10 seconds
    policy = fetch_policy(conn, policy_id)

    random_state = numpy.random.RandomState(env_seed)
    env = OneMaxOLL(n=n)

    for episode in range(num_evaluation_episodes):
      episode_seed = random_state.randint(100_000)
      episode_length = evaluate_episode(env, policy, episode_seed)

      # Save episode info to the database within the connection's context
      try:
        with conn:  # This automatically handles transactions
          cursor = conn.cursor()
          cursor.execute(
            'INSERT INTO episode_info (policy_id, episode_seed, episode_length, num_function_evaluations) VALUES (?, ?, ?, ?);',
            (policy_id, episode_seed, episode_length, int(env.num_function_evaluations)),
          )
      except sqlite3.OperationalError as e:
        # Handle database lock error
        print(f"Database lock error: {e}", file=sys.stderr)

def evaluate_episode(env, policy, episode_seed):
  """Simulate an episode based on the policy and return fitness at each step."""
  state = env.reset(episode_seed)[0]
  done = False
  fitness_values = []

  while not done:
    fitness_values.append(state)
    action = policy[state]
    next_state, _, done, _ = env.step(action)
    state = next_state[0]

  return len(fitness_values) - 1

def fetch_policy(conn, policy_id):
  """Fetch a policy from the database and reconstruct it as a dictionary."""
  cursor = conn.cursor()
  cursor.execute('SELECT fitness, lambda FROM policies_data WHERE policy_id = ?', (policy_id,))
  rows = cursor.fetchall()
  return {fitness: lambda_val for fitness, lambda_val in rows}

def insert_policy_info(conn, policy_id, num_training_episodes, num_q_table_updates):
  """Update policies_info with the number of training episodes."""
  with conn:
    conn.execute('UPDATE policies_info SET num_training_episodes = ? WHERE policy_id = ?;', (num_training_episodes, policy_id))
    conn.execute('UPDATE policies_info SET num_q_table_updates = ? WHERE policy_id = ?;', (num_q_table_updates, policy_id))

def drop_all_tables(db_path):

  if not os.path.exists(db_path): return

  """Drop all tables from the database and delete from sqlite_sequence if it exists."""
  with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    # Check if sqlite_sequence table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence';")
    if cursor.fetchone():
      cursor.execute("DELETE FROM sqlite_sequence;")

    # Retrieve a list of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    # Generate a script to drop all tables
    drop_script = "\n".join([f"DROP TABLE IF EXISTS {table[0]};" for table in tables if table[0] != "sqlite_sequence"])
    cursor.executescript(drop_script)

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

  # Insert and evaluate the special policy
  insert_special_policy(conn, config['n'])
  evaluate_policy(-1, config['db_path'], config['n'], config['env_seed'], config['num_evaluation_episodes'])

  # Q-learning process
  env = OneMaxOLL(n=config['n'])
  seed = numpy.random.randint(0, 100_000)



  # When initializing the database, pass the lock to the q_learning function
  q_table = q_learning_and_save_policy(
    env,
    config['episodes'],
    config['learning_rate'],
    config['gamma'],
    config['epsilon'],
    seed,
    conn,
    config['evaluation_interval'],
  )

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
