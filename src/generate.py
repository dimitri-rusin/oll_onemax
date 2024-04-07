from datetime import datetime
import ray
from ray.rllib.utils import check_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import Policy
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
  def __init__(
    self,
    n,
    seed=None,
    reward_type = 'ONLY_EVALUATIONS',
    action_type='DISCRETE',
    num_actions=None,
    state_type='ONE_HOT_ENCODED',
  ):
    super(OneMaxOLL, self).__init__()
    self.n = n
    assert num_actions is not None
    assert action_type in ['DISCRETE', 'CONTINUOUS']
    if action_type == 'DISCRETE':
      self.action_space = gymnasium.spaces.Discrete(num_actions)
    if action_type == 'CONTINUOUS':
      self.action_space = gymnasium.spaces.Box(low=0, high=num_actions - 1, shape=(1,), dtype=numpy.float64)

    if state_type == 'SCALAR_ENCODED':
      self.observation_space = gymnasium.spaces.Box(low=0, high=n, shape=(1,), dtype=numpy.int32)
      # high=n rather high=(n - 1), because the terminal observations are evaluated by RLlib
    if state_type == 'ONE_HOT_ENCODED':
      self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(n,), dtype=numpy.int32)
    self.state_type = state_type

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

    fitness_array = None
    if self.state_type == 'SCALAR_ENCODED':
      fitness_array = numpy.array([self.current_solution.fitness])
    if self.state_type == 'ONE_HOT_ENCODED':
      fitness_array = create_fitness_vector(self.current_solution.fitness)

    return fitness_array, {}

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

    fitness_array = None
    if self.state_type == 'SCALAR_ENCODED':
      fitness_array = numpy.array([self.current_solution.fitness])
    if self.state_type == 'ONE_HOT_ENCODED':
      fitness_array = create_fitness_vector(self.current_solution.fitness)

    return fitness_array, reward, terminated, truncated, info

# ============== ENVIRONMENT - END ==============

def create_fitness_vector(fitness):
  fitness_vector = numpy.zeros(config['n'], dtype=numpy.int32)
  if fitness < config['n']:
    fitness_vector[fitness] = 1
  return fitness_vector

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
    num_function_evaluations, num_evaluation_timesteps = evaluate_episode(policy, episode_seed)

    # Collect episode data
    episode_data.append((policy_id, episode_seed, num_evaluation_timesteps, num_function_evaluations))

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
  num_function_evaluations, num_evaluation_timesteps = onell_algs_rs.onell_lambda(
    n,
    policy_list,
    episode_seed,
    999_999_999,
    config['probability_of_closeness_to_optimum'],
  )

  return num_function_evaluations, num_evaluation_timesteps

def fetch_policy(database, policy_id):
  """Fetch a policy from the database and reconstruct it as a dictionary."""
  cursor = database.cursor()
  cursor.execute('SELECT fitness, lambda_minus_one FROM POLICY_DETAILS WHERE policy_id = ?', (policy_id,))
  rows = cursor.fetchall()
  return {fitness: lambda_val for fitness, lambda_val in rows}

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




def evaluate_policy_and_write_to_db(num_timesteps, policy):
  with sqlite3.connect(config['db_path']) as database:
    cursor = database.cursor()
    current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cursor.execute('INSERT INTO CONSTRUCTED_POLICIES (num_total_timesteps, created_at) VALUES (?, ?);', (num_timesteps, current_time))
    policy_id = cursor.lastrowid
    policy_data = [(int(policy_id), int(fitness), int(lambda_minus_one)) for fitness, lambda_minus_one in policy.items()]
    cursor.executemany('INSERT INTO POLICY_DETAILS (policy_id, fitness, lambda_minus_one) VALUES (?, ?, ?);', policy_data)

  seed_for_generating_episode_seeds = numpy.random.randint(999_999)
  episode_seed_generator = numpy.random.RandomState(seed_for_generating_episode_seeds)
  num_function_evaluations_list = []

  num_evaluation_episodes = config['num_evaluation_episodes']
  for episode_index in range(num_evaluation_episodes):
    print(f"Policy: Evaluating episode {episode_index + 1} / {num_evaluation_episodes}.")
    episode_seed = episode_seed_generator.randint(999_999)
    num_function_evaluations, num_evaluation_timesteps = evaluate_episode(policy, episode_seed)
    num_function_evaluations_list.append((policy_id, episode_seed, num_evaluation_timesteps, num_function_evaluations))

  with sqlite3.connect(config['db_path'], timeout=10) as database:
    cursor = database.cursor()
    cursor.executemany(
      'INSERT INTO EVALUATION_EPISODES (policy_id, episode_seed, episode_length, num_function_evaluations) VALUES (?, ?, ?, ?);',
      num_function_evaluations_list
    )

  only_fitness = [int(lambda_minus_one) for fitness, lambda_minus_one in policy.items()]
  print("policy", only_fitness)







def main():
  print("HALLO!")
  load_config()

  random_seed = config['random_seed']
  numpy.random.seed(random_seed)

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
  n = config['n']

  env_seed = numpy.random.randint(999_999)
  class OneMaxOLLWrapper(gymnasium.Env):
      def __init__(self, c=None):
          # Extract parameters from the config
          n = config['n']
          num_actions = config['num_lambdas']
          state_type = config['state_type']
          action_type = config['action_type']
          reward_type = config['reward_type']

          # Initialize the OneMaxOLL environment with the specified parameters
          self.env = OneMaxOLL(
              n=n,
              seed=env_seed,
              num_actions=num_actions,
              state_type=state_type,
              action_type=action_type,
              reward_type=reward_type,
          )

          # Set this wrapper's action and observation space to match the inner environment
          self.action_space = self.env.action_space
          self.observation_space = self.env.observation_space

      def reset(self, *, seed=None, options=None):
          return self.env.reset(seed)

      def step(self, action):
          return self.env.step(action)

      def render(self, mode='human'):
          return self.env.render(mode)

      def close(self):
          return self.env.close()

  import ray
  from ray.rllib.utils import check_env
  env = OneMaxOLLWrapper()
  check_env(env)

  from ray.rllib.algorithms.ppo import PPOConfig
  from ray.tune.logger import pretty_print
  algo = (
    PPOConfig()
    .rollouts(rollout_fragment_length=config['num_timesteps_per_evaluation'], num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env=OneMaxOLLWrapper)
    .build()
  )
  inspectify.d(algo)
  base_checkpoint_dir = "computed/ray/"
  num_training_iterations = 10
  for training_iteration in range(num_training_iterations):
    result = algo.train()
    print(pretty_print(result))
    checkpoint_dir = os.path.join(base_checkpoint_dir, str(training_iteration + 1))
    os.makedirs(checkpoint_dir, exist_ok=True)
    algo.save_checkpoint(checkpoint_dir)




    algo = PPOConfig().environment(env=OneMaxOLLWrapper).build()
    algo.restore(checkpoint_dir)

    # Generate all possible states
    n = config['n']  # Number of dimensions
    all_states = numpy.arange(n).reshape(-1, 1)

    # Predict actions for all states
    predicted_actions = []
    policy = {}
    for state in all_states:
      action = algo.compute_single_action(state)
      predicted_actions.append(action)
      policy[state[0]] = action



    inspectify.d(predicted_actions)
    inspectify.d(policy)
    inspectify.d(algo.config["rollout_fragment_length"])
    evaluate_policy_and_write_to_db(training_iteration + 1, policy)











    print(f"Checkpoint saved in directory {checkpoint_dir}")

if __name__ == '__main__':
  main()
