from datetime import datetime
import collections.abc
import datetime
import gymnasium
import inspectify
import numpy
import onell_algs_rs
import os
import sqlite3
import sys
import time
import time
import yaml

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import dacbench_adjustments.onell_algs

config = None





# ============== ENVIRONMENT - BEGIN ==============

class OneMaxOLL(gymnasium.Env):
  def __init__(self, n, seed=None):
    super(OneMaxOLL, self).__init__()
    self.n = n
    num_actions = int(numpy.sqrt(n))
    self.action_space = gymnasium.spaces.Discrete(num_actions)
    self.observation_space = gymnasium.spaces.Box(low=0, high=n - 1, shape=(1,), dtype=numpy.int32)
    self.seed = seed
    self.random = numpy.random.RandomState(self.seed)
    self.random_number_generator = numpy.random.default_rng(self.seed)
    assert seed is None
    self.current_solution = None
    self.num_function_evaluations = None
    self.num_total_timesteps = 0
    self.num_total_function_evaluations = 0

  def reset(self, episode_seed):
    # Use the provided seed for reproducibility
    self.seed = episode_seed
    self.num_function_evaluations = 0
    self.random = numpy.random.RandomState(self.seed)
    self.random_number_generator = numpy.random.default_rng(self.seed)

    initial_fitness = self.n
    while initial_fitness >= self.n:
      self.current_solution = dacbench_adjustments.onell_algs.OneMax(
        self.n,
        rng = self.random_number_generator,
        ratio_of_optimal_bits = config['probability_of_closeness_to_optimum'],
      )
      initial_fitness = self.current_solution.fitness

    self.num_function_evaluations += 1 # there is one evaluation [call to .eval()] inside OneMax

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
    self.num_total_function_evaluations += num_evaluations_of_this_step
    self.num_total_timesteps += 1

    return numpy.array([self.current_solution.fitness]), reward, terminated, info

# ============== ENVIRONMENT - END ==============



def q_learning_and_save_policy(learning_rate, gamma, epsilon, database, evaluation_interval):
  """Perform standard Q-learning, update Q-table, choose actions, save and evaluate policies."""

  training_environment = OneMaxOLL(n=config['n'])

  # Initialize a Q-table with zeros
  num_observations = training_environment.observation_space.high[0] - training_environment.observation_space.low[0] + 1
  q_table = numpy.zeros((num_observations, training_environment.action_space.n))

  max_training_timesteps = config['max_training_timesteps']
  num_q_table_updates = 0
  num_training_episodes = 0
  num_training_timesteps = 1
  sum_initial_fitness = 0
  sum_squares_initial_fitness = 0

  policy = numpy.argmax(q_table, axis=1)
  policy_id = insert_policy_and_get_id(database, policy)
  insert_policy_info(database, policy_id, training_environment.num_total_timesteps, 0, training_environment.num_total_function_evaluations, 0, 0)
  seed_for_generating_episode_seeds = numpy.random.randint(100_000)
  evaluate_policy(policy_id, config['db_path'], config['n'], seed_for_generating_episode_seeds, config['num_evaluation_episodes'])

  while num_training_timesteps < max_training_timesteps:
    print(f"Training timestep {num_training_timesteps:,} / {max_training_timesteps:,}.")
    episode_seed = numpy.random.randint(100_000)
    fitness, done = training_environment.reset(episode_seed)[0], False

    # Update sum and sum of squares for initial fitness
    num_training_episodes += 1
    sum_initial_fitness += fitness
    sum_squares_initial_fitness += fitness ** 2

    while not done:
      # Choose an action based on the current fitness and Q-table (Epsilon-greedy strategy)
      if numpy.random.random() < epsilon:
        action = numpy.random.randint(training_environment.action_space.n)
      else:
        action = numpy.argmax(q_table[fitness])

      # Perform the action
      next_fitness, reward, done, _ = training_environment.step(action)
      num_training_timesteps += 1

      # Update the Q-table using the Q-learning algorithm
      q_predict = q_table[fitness, action]
      q_target = reward if done else reward + gamma * numpy.max(q_table[next_fitness[0]])
      q_table[fitness, action] += learning_rate * (q_target - q_predict)
      num_q_table_updates += 1

      fitness = next_fitness[0]

    # Evaluate policy at specified intervals
    if num_training_episodes % evaluation_interval == 0:
      seed_for_generating_episode_seeds = numpy.random.randint(100_000)
      evaluate_policy_and_write_to_database(num_training_episodes, sum_initial_fitness, sum_squares_initial_fitness, q_table, database, training_environment, seed_for_generating_episode_seeds)

  seed_for_generating_episode_seeds = numpy.random.randint(100_000)
  evaluate_policy_and_write_to_database(num_training_episodes, sum_initial_fitness, sum_squares_initial_fitness, q_table, database, training_environment, seed_for_generating_episode_seeds)

  return q_table

def evaluate_policy_and_write_to_database(num_training_episodes, sum_initial_fitness, sum_squares_initial_fitness, q_table, database, training_environment, seed_for_generating_episode_seeds):
  mean_initial_fitness = sum_initial_fitness / num_training_episodes
  variance_initial_fitness = (sum_squares_initial_fitness / num_training_episodes) - (mean_initial_fitness ** 2)

  policy = numpy.argmax(q_table, axis=1)
  policy_id = insert_policy_and_get_id(database, policy)
  insert_policy_info(
    database,
    policy_id,
    training_environment.num_total_timesteps,
    num_training_episodes,
    training_environment.num_total_function_evaluations,
    mean_initial_fitness,
    variance_initial_fitness,
  )
  evaluate_policy(policy_id, config['db_path'], config['n'], seed_for_generating_episode_seeds, config['num_evaluation_episodes'])

def create_tables(database):
    """Create necessary tables in the database."""
    with database:
        database.executescript('''
          CREATE TABLE IF NOT EXISTS POLICY_DETAILS (policy_id INTEGER, fitness INTEGER, lambda_minus_one INTEGER);
          CREATE TABLE IF NOT EXISTS CONSTRUCTED_POLICIES (policy_id INTEGER PRIMARY KEY AUTOINCREMENT, num_total_timesteps INTEGER, num_training_episodes INTEGER, num_total_function_evaluations INTEGER, mean_initial_fitness DOUBLE, variance_initial_fitness DOUBLE, FOREIGN KEY(policy_id) REFERENCES POLICY_DETAILS(policy_id));
          CREATE TABLE IF NOT EXISTS EVALUATION_EPISODES (policy_id INTEGER, episode_id INTEGER PRIMARY KEY AUTOINCREMENT, episode_seed INTEGER, episode_length INTEGER, num_function_evaluations INTEGER, FOREIGN KEY(policy_id) REFERENCES POLICY_DETAILS(policy_id));
          CREATE TABLE IF NOT EXISTS CONFIG (key TEXT PRIMARY KEY, value TEXT);
        ''')

def insert_config(database, config):
    """Insert config dictionary into CONFIG table."""
    with database:
        for key, value in config.items():
            database.execute('INSERT INTO CONFIG (key, value) VALUES (?, ?)', (key, str(value)))

def insert_special_policy(database, num_dimensions):
  """Insert the special policy with policy_id -1."""

  policy_lambdas = [int(numpy.sqrt(num_dimensions / (num_dimensions - fitness))) - 1 for fitness in range(num_dimensions)]
  # - 1, because the environment expects lambda - 1.

  policy_id = -1  # Special policy ID
  insert_policy_and_get_id(database, policy_lambdas, policy_id)

def insert_policy_and_get_id(database, policy, policy_id=None):
  """Insert policy into POLICY_DETAILS and return the generated policy_id."""
  with database:  # This automatically begins and commits/rollbacks a transaction
    cursor = database.cursor()
    if policy_id is None:
      cursor.execute('INSERT INTO CONSTRUCTED_POLICIES (num_training_episodes) VALUES (?);', (0,))
      policy_id = cursor.lastrowid
    else:
      cursor.execute('INSERT INTO CONSTRUCTED_POLICIES (policy_id, num_total_timesteps, num_training_episodes, num_total_function_evaluations, mean_initial_fitness, variance_initial_fitness) VALUES (?, ?, ?, ?, ?, ?);', (policy_id, 0, 0, 0, 0, 0))
    cursor.executemany('INSERT INTO POLICY_DETAILS (policy_id, fitness, lambda_minus_one) VALUES (?, ?, ?);',
                       [(policy_id, int(fitness), int(lambda_minus_one)) for fitness, lambda_minus_one in enumerate(policy)])
  return policy_id

def evaluate_policy(policy_id, db_path, n, seed_for_generating_episode_seeds, num_evaluation_episodes):
  """Evaluate policy using multiple processes."""
  policy = fetch_policy(sqlite3.connect(db_path), policy_id)

  episode_seed_generator = numpy.random.RandomState(seed_for_generating_episode_seeds)
  episode_data = []  # List to store episode data

  for episode_index in range(num_evaluation_episodes):
    print(f"Policy {policy_id}: Evaluating episode {episode_index + 1} / {num_evaluation_episodes}.")
    episode_seed = episode_seed_generator.randint(100_000)
    num_function_evaluations = evaluate_episode(policy, episode_seed)

    # Collect episode data
    episode_data.append((policy_id, episode_seed, None, num_function_evaluations))

  # Write all collected data to the database in a single transaction
  with sqlite3.connect(db_path, timeout=10) as database:
    with database:  # This automatically handles transactions
      cursor = database.cursor()
      cursor.executemany(
        'INSERT INTO EVALUATION_EPISODES (policy_id, episode_seed, episode_length, num_function_evaluations) VALUES (?, ?, ?, ?);',
        episode_data
      )

def evaluate_episode(policy, episode_seed):
  """Simulate an episode based on the policy and return fitness at each step."""

  policy_list = [policy[fitness] for fitness in policy]
  for i in range(len(policy_list)):
    policy_list[i] += 1

  num_function_evaluations = onell_algs_rs.onell_lambda(
    config['n'],
    policy_list,
    episode_seed,
    999_999_999,
    config['probability_of_closeness_to_optimum'],
  )

  return num_function_evaluations

def fetch_policy(database, policy_id):
  """Fetch a policy from the database and reconstruct it as a dictionary."""
  cursor = database.cursor()
  cursor.execute('SELECT fitness, lambda_minus_one FROM POLICY_DETAILS WHERE policy_id = ?', (policy_id,))
  rows = cursor.fetchall()
  return {fitness: lambda_val for fitness, lambda_val in rows}

def insert_policy_info(database, policy_id, num_total_timesteps, num_training_episodes, num_total_function_evaluations, mean_initial_fitness, variance_initial_fitness):
  with database:
    database.execute('UPDATE CONSTRUCTED_POLICIES SET num_training_episodes = ? WHERE policy_id = ?;', (num_training_episodes, policy_id))
    database.execute('UPDATE CONSTRUCTED_POLICIES SET num_total_function_evaluations = ? WHERE policy_id = ?;', (num_total_function_evaluations, policy_id))
    database.execute('UPDATE CONSTRUCTED_POLICIES SET num_total_timesteps = ? WHERE policy_id = ?;', (num_total_timesteps, policy_id))
    database.execute('UPDATE CONSTRUCTED_POLICIES SET mean_initial_fitness = ? WHERE policy_id = ?;', (mean_initial_fitness, policy_id))
    database.execute('UPDATE CONSTRUCTED_POLICIES SET variance_initial_fitness = ? WHERE policy_id = ?;', (variance_initial_fitness, policy_id))

def drop_all_tables(db_path):

  if not os.path.exists(db_path): return

  """Drop all tables from the database and delete from sqlite_sequence if it exists."""
  with sqlite3.connect(db_path) as database:
    cursor = database.cursor()
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
  global config
  config = {}

  for key, value in os.environ.items():
    if key.startswith("OO__"):
      # Remove 'OO__' prefix and convert to lowercase
      key = key[4:].lower()

      # Split the key at double underscores
      key_parts = key.split('__')

      # Infer the type
      if value.isdigit():
        parsed_value = int(value)
      elif all(char.isdigit() or char == '.' for char in value):
        try:
          parsed_value = float(value)
        except ValueError:
          parsed_value = value
      else:
        parsed_value = value

      # Create nested dictionaries as necessary
      d = config
      for part in key_parts[:-1]:
        if part not in d:
          d[part] = {}
        d = d[part]
      d[key_parts[-1]] = parsed_value

  current_time = datetime.datetime.now()
  config['experiment_start_date'] = current_time.strftime("%Y-%B-%d %H:%M:%S") + " " + time.tzname[0]

  setup_config(config)
  database = setup_database(config['db_path'])
  create_tables(database)

  # Insert config into CONFIG table
  insert_config(database, flatten_dict(config))

  seed_for_generating_episode_seeds = numpy.random.randint(100_000)

  # Insert and evaluate the special policy
  insert_special_policy(database, config['n'])
  evaluate_policy(
    -1,
    config['db_path'],
    config['n'],
    seed_for_generating_episode_seeds,
    config['num_evaluation_episodes'],
  )

  # When initializing the database, pass the lock to the q_learning function
  q_table = q_learning_and_save_policy(
    config['learning_rate'],
    config['gamma'],
    config['epsilon'],
    database,
    config['evaluation_interval'],
  )

  database.close()

def flatten_dict(d, parent_key='', sep='__'):
    """
    Flatten a nested dictionary.
    Example: {'a': {'b': 1}} becomes {'a__b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def setup_config(config):
  """Setup global configuration parameters."""
  numpy.random.seed(config['random_seed'])

def setup_database(db_path):
  """Prepare the database, creating necessary directories and tables."""
  drop_all_tables(db_path)

  directory_path = os.path.dirname(db_path)
  if not os.path.exists(directory_path):
    os.makedirs(directory_path)

  return sqlite3.connect(db_path)

if __name__ == '__main__':
  main()
