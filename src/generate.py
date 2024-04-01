from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy as ep
from stable_baselines3.common.utils import set_random_seed
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
import torch
import yaml

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import dacbench_adjustments.onell_algs

config = None

# ============== ENVIRONMENT - BEGIN ==============

class OneMaxOLL(gymnasium.Env):
  def __init__(self, n, seed=None, reward_type = 'ONLY_EVALUATIONS', action_space_type='DISCRETE', num_actions=None):
    super(OneMaxOLL, self).__init__()
    self.n = n
    assert num_actions is not None
    assert action_space_type in ['DISCRETE', 'CONTINUOUS']
    if action_space_type == 'DISCRETE':
      self.action_space = gymnasium.spaces.Discrete(num_actions)
    if action_space_type == 'CONTINUOUS':
      self.action_space = gymnasium.spaces.Box(low=0, high=num_actions - 1, shape=(1,), dtype=numpy.float64)

    self.observation_space = gymnasium.spaces.Box(low=0, high=n - 1, shape=(1,), dtype=numpy.int32)
    self.seed = seed
    self.random = numpy.random.RandomState(self.seed)
    self.random_number_generator = numpy.random.default_rng(self.seed)
    self.current_solution = None
    self.num_function_evaluations = None
    self.num_total_timesteps = 0
    self.num_total_function_evaluations = 0
    self.reward_type = reward_type

  def reset(self, seed = None):
    # Use the provided seed for reproducibility
    if seed is not None:
      self.seed = seed
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

    return numpy.array([self.current_solution.fitness]), {}

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
    assert self.reward_type in ['ONLY_EVALUATIONS', 'EVALUATIONS_PLUS_FITNESS']
    if self.reward_type == 'ONLY_EVALUATIONS':
      reward = -num_evaluations_of_this_step
    if self.reward_type == 'EVALUATIONS_PLUS_FITNESS':
      reward = -num_evaluations_of_this_step + (self.current_solution.fitness - prior_fitness)
    terminated = self.current_solution.is_optimal()
    info = {}

    self.num_function_evaluations += num_evaluations_of_this_step
    self.num_total_function_evaluations += num_evaluations_of_this_step
    self.num_total_timesteps += 1

    truncated = False

    return numpy.array([self.current_solution.fitness]), reward, terminated, truncated, info

# ============== ENVIRONMENT - END ==============

def create_tables(database):
  """Create necessary tables in the database."""
  with database:
    database.executescript('''
      CREATE TABLE IF NOT EXISTS POLICY_DETAILS (policy_id INTEGER, fitness INTEGER, lambda_minus_one INTEGER);

      CREATE TABLE IF NOT EXISTS CONSTRUCTED_POLICIES (
        policy_id INTEGER PRIMARY KEY AUTOINCREMENT,
        num_total_timesteps INTEGER,
        num_training_episodes INTEGER,
        num_total_function_evaluations INTEGER,
        mean_initial_fitness DOUBLE,
        variance_initial_fitness DOUBLE,
        created_at TEXT, -- ISO8601 format: 'YYYY-MM-DDTHH:MM:SS.SSSZ'
        FOREIGN KEY(policy_id) REFERENCES POLICY_DETAILS(policy_id)
      );

      CREATE TABLE IF NOT EXISTS EVALUATION_EPISODES (policy_id INTEGER, episode_id INTEGER PRIMARY KEY AUTOINCREMENT, episode_seed INTEGER, episode_length INTEGER, num_function_evaluations INTEGER, FOREIGN KEY(policy_id) REFERENCES POLICY_DETAILS(policy_id));
      CREATE TABLE IF NOT EXISTS CONFIG (key TEXT PRIMARY KEY, value TEXT);
    ''')

def insert_config(database, config):
  """Insert config dictionary into CONFIG table."""
  with database:
    for key, value in config.items():
      database.execute('INSERT INTO CONFIG (key, value) VALUES (?, ?)', (key, str(value)))

def insert_theory_derived_policy(database, num_dimensions):
  """Insert the theory-derived policy with policy_id -1."""
  policy_lambdas = [int(numpy.sqrt(num_dimensions / (num_dimensions - fitness))) - 1 for fitness in range(num_dimensions)]
  # - 1, because the environment expects lambda - 1.
  policy_id = -1  # Theory-derived policy ID
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
    episode_seed = episode_seed_generator.randint(999_999)
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

  policy_list = []
  for _, λ in policy.items():
    if isinstance(λ, numpy.ndarray) and λ.size == 1:
      λ = λ.item()
    λ += 1
    policy_list.append(λ)

  n = len(policy)
  num_function_evaluations = onell_algs_rs.onell_lambda(
    n,
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

def setup_database(db_path):
  """Prepare the database, creating necessary directories and tables."""
  drop_all_tables(db_path)

  directory_path = os.path.dirname(db_path)
  if not os.path.exists(directory_path):
    os.makedirs(directory_path)

  return sqlite3.connect(db_path)

def load_config():
  global config
  config = {}
  for key, value in os.environ.items():
    if key.startswith("OO__"):
      # Remove 'OO__' prefix and convert to lowercase
      key = key[4:].lower()

      # Split the key at double underscores
      key_parts = key.split('__')

      # Check for list-like structure and parse accordingly
      if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]  # Remove the brackets
        parsed_value = []
        for item in value.split(','):
          item = item.strip()  # Remove whitespace
          if item.isdigit():
            parsed_value.append(int(item))
          elif all(char.isdigit() or char == '.' for char in item):
            try:
              parsed_value.append(float(item))
            except ValueError:
              parsed_value.append(item)
          else:
            parsed_value.append(item)
      elif value.isdigit():
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

def main():

  load_config()

  numpy.random.seed(config['random_seed'])

  database = setup_database(config['db_path'])
  create_tables(database)
  insert_config(database, config)

  seed_for_generating_episode_seeds = numpy.random.randint(999_999)
  insert_theory_derived_policy(database, config['n'])
  evaluate_policy(
    -1,
    config['db_path'],
    config['n'],
    seed_for_generating_episode_seeds,
    config['num_evaluation_episodes'],
  )

  num_evaluation_episodes = config['num_evaluation_episodes']

  class PPOCallback(BaseCallback):
    def __init__(self, verbose=0):
      super(PPOCallback, self).__init__(verbose)
      self.evaluation_results = []

    def _on_step(self):
      if self.num_timesteps % config['num_timesteps_per_evaluation'] == 0:
        policy = {obs_value: self.model.predict(numpy.array([obs_value]).reshape((1, 1)), deterministic=True)[0][0] for obs_value in range(n)}
        with sqlite3.connect(config['db_path']) as database:
          cursor = database.cursor()
          current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
          cursor.execute('INSERT INTO CONSTRUCTED_POLICIES (num_total_timesteps, created_at) VALUES (?, ?);', (self.num_timesteps, current_time))
          policy_id = cursor.lastrowid
          policy_data = [(int(policy_id), int(fitness), int(lambda_minus_one)) for fitness, lambda_minus_one in policy.items()]
          cursor.executemany('INSERT INTO POLICY_DETAILS (policy_id, fitness, lambda_minus_one) VALUES (?, ?, ?);', policy_data)

        seed_for_generating_episode_seeds = numpy.random.randint(999_999)
        episode_seed_generator = numpy.random.RandomState(seed_for_generating_episode_seeds)
        num_function_evaluations_list = []

        for episode_index in range(num_evaluation_episodes):
          print(f"Policy: Evaluating episode {episode_index + 1} / {num_evaluation_episodes}.")
          episode_seed = episode_seed_generator.randint(999_999)
          num_function_evaluations = evaluate_episode(policy, episode_seed)
          num_function_evaluations_list.append((policy_id, episode_seed, None, num_function_evaluations))

        with sqlite3.connect(config['db_path'], timeout=10) as database:
          cursor = database.cursor()
          cursor.executemany(
            'INSERT INTO EVALUATION_EPISODES (policy_id, episode_seed, episode_length, num_function_evaluations) VALUES (?, ?, ?, ?);',
            num_function_evaluations_list
          )

        only_fitness = [int(lambda_minus_one) for fitness, lambda_minus_one in policy.items()]
        print("policy", only_fitness)

      return True

  n = config['n']
  ppo_seed = numpy.random.randint(999_999)
  set_random_seed(ppo_seed)

  ppo_agent = PPO(
    config['ppo']['policy'],
    OneMaxOLL(
      n = n,
      seed = ppo_seed,
      reward_type = config['reward_type'],
      action_space_type = config['action_space_type'],
      num_actions = config['num_lambdas'],
    ),
    policy_kwargs = {'net_arch': config['ppo']['net_arch'], 'activation_fn': torch.nn.ReLU},
    learning_rate = config['ppo']['learning_rate'],
    n_steps = config['ppo']['n_steps'],
    batch_size = config['ppo']['batch_size'],
    n_epochs = config['ppo']['n_epochs'],
    gamma = config['ppo']['gamma'],
    gae_lambda = config['ppo']['gae_lambda'],
    vf_coef = config['ppo']['vf_coef'],
    ent_coef = config['ppo']['ent_coef'],
    clip_range = config['ppo']['clip_range'],
    verbose = 1,
  )

  policy = {obs_value: ppo_agent.predict(numpy.array([obs_value]).reshape((1, 1)), deterministic=True)[0][0] for obs_value in range(n)}
  with sqlite3.connect(config['db_path']) as database:
    cursor = database.cursor()
    current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cursor.execute('INSERT INTO CONSTRUCTED_POLICIES (num_total_timesteps, created_at) VALUES (?, ?);', (0, current_time))
    policy_id = cursor.lastrowid
    policy_data = [(int(policy_id), int(fitness), int(lambda_minus_one)) for fitness, lambda_minus_one in policy.items()]
    cursor.executemany('INSERT INTO POLICY_DETAILS (policy_id, fitness, lambda_minus_one) VALUES (?, ?, ?);', policy_data)

  seed_for_generating_episode_seeds = numpy.random.randint(999_999)
  episode_seed_generator = numpy.random.RandomState(seed_for_generating_episode_seeds)
  num_function_evaluations_list = []

  for episode_index in range(num_evaluation_episodes):
    print(f"Policy: Evaluating episode {episode_index + 1} / {num_evaluation_episodes}.")
    episode_seed = episode_seed_generator.randint(999_999)
    num_function_evaluations = evaluate_episode(policy, episode_seed)
    num_function_evaluations_list.append((policy_id, episode_seed, None, num_function_evaluations))

  with sqlite3.connect(config['db_path'], timeout=10) as database:
    cursor = database.cursor()
    cursor.executemany(
      'INSERT INTO EVALUATION_EPISODES (policy_id, episode_seed, episode_length, num_function_evaluations) VALUES (?, ?, ?, ?);',
      num_function_evaluations_list
    )

  only_fitness = [int(lambda_minus_one) for fitness, lambda_minus_one in policy.items()]
  print("policy", only_fitness)

  ppo_agent.learn(total_timesteps=config['max_training_timesteps'], callback=PPOCallback())



if __name__ == '__main__':
  main()
