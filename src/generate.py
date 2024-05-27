import datetime
import gymnasium
import inspectify
import numpy
import onell_algs_rs
import os
import sqlite3
import stable_baselines3.common.env_util
import stable_baselines3.common.utils
import torch

import pathlib
import sys
import_path = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.append(import_path)
import DE0CH_OLL.tuned_with_irace.onell_algs

# ============== ENVIRONMENT - BEGIN ==============

class OneMaxOLL(gymnasium.Env):
  def __init__(self, dimensionality, seed, reward_type, action_type, state_type, config):
    super(OneMaxOLL, self).__init__()
    self.dimensionality = dimensionality
    self.config = config

    if 'lambdas' in self.config:
      self.lambdas = numpy.array(self.config['lambdas'], dtype = numpy.float64)
      self.action_space = gymnasium.spaces.Discrete(self.lambdas.shape[0])

    if 'mutation_rates' in self.config:
      self.mutation_rates = numpy.array(self.config['mutation_rates'], dtype = numpy.float64)
      self.mutation_sizes = numpy.array(self.config['mutation_sizes'], dtype = numpy.int64)
      self.crossover_rates = numpy.array(self.config['crossover_rates'], dtype = numpy.float64)
      self.crossover_sizes = numpy.array(self.config['crossover_sizes'], dtype = numpy.int64)
      self.action_space = gymnasium.spaces.MultiDiscrete([
        self.mutation_rates.shape[0],
        self.mutation_sizes.shape[0],
        self.crossover_rates.shape[0],
        self.crossover_sizes.shape[0],
      ])

    if state_type == 'SCALAR_ENCODED':
      self.observation_space = gymnasium.spaces.Box(low=0, high=dimensionality - 1, shape=(1,), dtype=numpy.int32)
    if state_type == 'ONE_HOT_ENCODED':
      self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(dimensionality,), dtype=numpy.int32)
    self.state_type = state_type

    self.seed = seed
    self.random_number_generator = numpy.random.Generator(numpy.random.MT19937(self.seed))
    self.current_solution = None
    self.num_function_evaluations = None
    self.num_total_timesteps = 0
    self.num_total_function_evaluations = 0
    self.reward_type = reward_type

  def reset(self, seed = None):
    if seed is not None:
      self.seed = seed
    self.num_function_evaluations = 0
    self.random_number_generator = numpy.random.Generator(numpy.random.MT19937(self.seed))

    self.current_fitness = self.dimensionality
    while self.current_fitness >= self.dimensionality:
      ratio_of_optimal_bits = self.config['closeness_to_optimum']
      self.current_bitstring = self.random_number_generator.choice(
        [True, False],
        size = self.dimensionality,
        p = [ratio_of_optimal_bits, 1 - ratio_of_optimal_bits],
      )
      self.current_fitness = self.current_bitstring.sum()

    self.num_function_evaluations += 1

    fitness_array = None
    if self.state_type == 'SCALAR_ENCODED':
      fitness_array = numpy.array([self.current_fitness])
    if self.state_type == 'ONE_HOT_ENCODED':
      fitness_array = create_fitness_vector(self.current_fitness, self.dimensionality)

    return fitness_array, {}

  def step(self, action_index):

    if 'lambdas' in self.config:
      # action_index is of type: <class 'numpy.int64'>
      lambda_ = self.lambdas[action_index]
      mutation_rate = numpy.float64(lambda_ / self.dimensionality)
      mutation_size = numpy.int64(lambda_)
      crossover_rate = numpy.float64(1.0 / lambda_)
      crossover_size = numpy.int64(lambda_)

    if 'mutation_rates' in self.config:
      # action_index is of type: <class 'numpy.ndarray'> (with 4 elements)
      mutation_rate = self.mutation_rates[action_index[0]]
      mutation_size = self.mutation_sizes[action_index[1]]
      crossover_rate = self.crossover_rates[action_index[2]]
      crossover_size = self.crossover_sizes[action_index[3]]

    generation_seed = self.random_number_generator.integers(int(1e9))

    next_bitstring, num_function_evaluations_of_this_step = onell_algs_rs.generation_full_py(
      self.current_bitstring.tolist(),
      mutation_rate,
      mutation_size,
      crossover_rate,
      crossover_size,
      generation_seed,
    )

    prior_fitness = self.current_fitness

    self.current_bitstring = numpy.array(next_bitstring, dtype=bool)
    self.current_fitness = self.current_bitstring.sum()

    assert self.reward_type in ['ONLY_EVALUATIONS', 'EVALUATIONS_PLUS_FITNESS']
    if self.reward_type == 'ONLY_EVALUATIONS':
      reward = -num_function_evaluations_of_this_step
    if self.reward_type == 'EVALUATIONS_PLUS_FITNESS':
      reward = -num_function_evaluations_of_this_step + (self.current_fitness - prior_fitness)

    terminated = (self.current_fitness == self.dimensionality)
    info = {}

    self.num_function_evaluations += num_function_evaluations_of_this_step
    self.num_total_function_evaluations += num_function_evaluations_of_this_step
    self.num_total_timesteps += 1

    truncated = False

    fitness_array = None
    if self.state_type == 'SCALAR_ENCODED':
      fitness_array = numpy.array([self.current_fitness])
    if self.state_type == 'ONE_HOT_ENCODED':
      fitness_array = create_fitness_vector(self.current_fitness, self.dimensionality)

    return fitness_array, reward, terminated, truncated, info

# ============== ENVIRONMENT - END ==============

def evaluate_episode(size_policy, episode_seed, closeness_to_optimum):

  oll_parameters = [parameter_tuple for _, parameter_tuple in size_policy.items()]
  dimensionality = len(size_policy)
  num_function_evaluations, num_evaluation_timesteps = onell_algs_rs.onell_lambda(
    dimensionality,
    oll_parameters,
    episode_seed,
    int(1e9),
    closeness_to_optimum,
  )

  return num_function_evaluations, num_evaluation_timesteps

def create_fitness_vector(fitness, dimensionality):
  fitness_vector = numpy.zeros(dimensionality, dtype=numpy.int32)
  if fitness < dimensionality:
    fitness_vector[fitness] = 1
  return fitness_vector

def load_config():
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
  return config

def flatten_config(prefix, nested_config):
  items = []
  for key, value in nested_config.items():
    new_key = f"{prefix}__{key}" if prefix else key
    if isinstance(value, dict):
      items.extend(flatten_config(new_key, value))
    else:
      items.append((new_key, value))
  return items

def evaluate_policy(
  crossover_rates,
  crossover_sizes,
  db_path,
  dimensionality,
  lambdas,
  mutation_rates,
  mutation_sizes,
  num_evaluation_episodes,
  num_training_timesteps,
  ppo_agent,
  state_type,
  closeness_to_optimum,
  episode_seed_generator,
):
  if state_type == 'ONE_HOT_ENCODED':
    index_policy = {int(obs_value): ppo_agent.predict(create_fitness_vector(obs_value, dimensionality), deterministic=True)[0].tolist() for obs_value in range(dimensionality)}
  if state_type == 'SCALAR_ENCODED':
    index_policy = {int(obs_value): ppo_agent.predict(numpy.array([obs_value]).reshape((1, 1)), deterministic=True)[0].tolist() for obs_value in range(dimensionality)}

  size_policy = {}

  if lambdas is not None:
    for fitness, lambda_index in index_policy.items():
      lambda_ = lambdas[lambda_index]
      mutation_rate = lambda_ / dimensionality
      crossover_rate = 1.0 / lambda_
      size_policy[fitness] = (numpy.float64(mutation_rate), numpy.int64(lambda_), numpy.float64(crossover_rate), numpy.int64(lambda_))

  if mutation_rates is not None:
    for fitness, (mutation_rate_index, mutation_size_index, crossover_rate_index, crossover_size_index) in index_policy.items():
      size_policy[fitness] = (mutation_rates[mutation_rate_index], int(mutation_sizes[mutation_size_index]), crossover_rates[crossover_rate_index], int(crossover_sizes[crossover_size_index]))

  with sqlite3.connect(db_path) as database:
    cursor = database.cursor()
    current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cursor.execute('INSERT INTO CONSTRUCTED_POLICIES (num_total_timesteps, created_at) VALUES (?, ?)', (num_training_timesteps, current_time))
    policy_id = cursor.lastrowid
    cursor.executemany(
      'INSERT INTO POLICY_DETAILS (policy_id, fitness, mutation_rate, mutation_size, crossover_rate, crossover_size) VALUES (?, ?, ?, ?, ?, ?)',
      [(policy_id, fitness, mutation_rate, mutation_size, crossover_rate, crossover_size) for fitness, (mutation_rate, mutation_size, crossover_rate, crossover_size) in size_policy.items()]
    )

  num_function_evaluations_list = []

  for episode_index in range(num_evaluation_episodes):
    print(f"Policy {policy_id}: Evaluating episode {episode_index + 1:,} / {num_evaluation_episodes:,}.")
    episode_seed = episode_seed_generator.integers(int(1e9))
    num_function_evaluations, num_evaluation_timesteps = evaluate_episode(size_policy, episode_seed, closeness_to_optimum)
    num_function_evaluations_list.append((policy_id, episode_seed, num_evaluation_timesteps, num_function_evaluations))

  with sqlite3.connect(db_path, timeout=10) as database:
    cursor = database.cursor()
    cursor.executemany(
      'INSERT INTO EVALUATION_EPISODES (policy_id, episode_seed, num_evaluation_timesteps, num_function_evaluations) VALUES (?, ?, ?, ?)',
      num_function_evaluations_list
    )















def main():
  config = load_config()

  main_generator_seed = config['random_seed']
  mersenne_twister = numpy.random.MT19937(main_generator_seed)
  main_generator = numpy.random.Generator(mersenne_twister)

  if os.path.isfile(config['db_path']):
    os.remove(config['db_path'])

  directory_path = os.path.dirname(config['db_path'])
  if not os.path.exists(directory_path):
    os.makedirs(directory_path, exist_ok=True)

  database = sqlite3.connect(config['db_path'])

  with database:
    database.executescript('''
      CREATE TABLE IF NOT EXISTS POLICY_DETAILS (policy_id INTEGER, fitness INTEGER, mutation_rate REAL, mutation_size INTEGER, crossover_rate REAL, crossover_size INTEGER);

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

      CREATE TABLE IF NOT EXISTS EVALUATION_EPISODES (policy_id INTEGER, episode_seed INTEGER, num_evaluation_timesteps INTEGER, num_function_evaluations INTEGER, FOREIGN KEY(policy_id) REFERENCES POLICY_DETAILS(policy_id));
      CREATE TABLE IF NOT EXISTS CONFIG (key TEXT PRIMARY KEY, value TEXT);
    ''')

  flattened_config = flatten_config('', config)

  with database:
    for key, value in flattened_config:
      database.execute('INSERT INTO CONFIG (key, value) VALUES (?, ?)', (key, str(value)))

  dimensionality = config['dimensionality']
  policy_id = -1
  theory_derived_size_policy = {fitness : int(numpy.sqrt(dimensionality / (dimensionality - fitness))) for fitness in range(dimensionality)}
  theory_derived_size_policy = {int(fitness): (lambda_ / dimensionality, lambda_, 1. / lambda_, lambda_) for fitness, lambda_ in theory_derived_size_policy.items()}
  with database:
    cursor = database.cursor()
    if policy_id is None:
      cursor.execute('INSERT INTO CONSTRUCTED_POLICIES (num_training_episodes) VALUES (?)', (0,))
      policy_id = cursor.lastrowid
    else:
      cursor.execute(
        'INSERT INTO CONSTRUCTED_POLICIES '
        '(policy_id, num_total_timesteps, num_training_episodes, '
        'num_total_function_evaluations, mean_initial_fitness, variance_initial_fitness) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (policy_id, 0, 0, 0, 0, 0)
      )
      cursor.executemany(
        'INSERT INTO POLICY_DETAILS (policy_id, fitness, mutation_rate, mutation_size, crossover_rate, crossover_size) VALUES (?, ?, ?, ?, ?, ?)',
        [(policy_id, fitness, mutation_rate, mutation_size, crossover_rate, crossover_size) for fitness, (mutation_rate, mutation_size, crossover_rate, crossover_size) in theory_derived_size_policy.items()]
      )

  db_path = config['db_path']
  num_evaluation_episodes = config['num_evaluation_episodes']


  episode_data = []
  for episode_index in range(num_evaluation_episodes):
    print(f"Policy {policy_id}: Evaluating episode {episode_index + 1:,} / {num_evaluation_episodes:,}.")
    episode_seed = main_generator.integers(int(1e9))
    num_function_evaluations, num_evaluation_timesteps = evaluate_episode(theory_derived_size_policy, episode_seed, config['closeness_to_optimum'])
    episode_data.append((policy_id, episode_seed, num_evaluation_timesteps, num_function_evaluations))

  with sqlite3.connect(db_path, timeout=10) as database:
    with database:
      cursor = database.cursor()
      cursor.executemany(
        'INSERT INTO EVALUATION_EPISODES (policy_id, episode_seed, num_evaluation_timesteps, num_function_evaluations) VALUES (?, ?, ?, ?)',
        episode_data
      )

  # hier nach lambda versus mutation_rates unterscheiden
  lambdas = None
  mutation_rates = None
  mutation_sizes = None
  crossover_rates = None
  crossover_sizes = None

  assert 'lambdas' in config or 'mutation_rates' in config

  if 'lambdas' in config:
    lambdas = numpy.array(config['lambdas'], dtype = numpy.float64)

  if 'mutation_rates' in config:
    mutation_rates = numpy.array(config['mutation_rates'], dtype = numpy.float64)
    mutation_sizes = numpy.array(config['mutation_sizes'], dtype = numpy.int64)
    crossover_rates = numpy.array(config['crossover_rates'], dtype = numpy.float64)
    crossover_sizes = numpy.array(config['crossover_sizes'], dtype = numpy.int64)

  class PPOCallback(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self, verbose=0):
      super(PPOCallback, self).__init__(verbose)
      self.evaluation_results = []

    def _on_step(self):

      # self.num_timesteps is the number of timesteps made across all environments, for each environment we have made n_steps steps.
      if self.num_timesteps % config['num_timesteps_per_evaluation'] != 0: return True

      evaluate_policy(
        crossover_rates = crossover_rates,
        crossover_sizes = crossover_sizes,
        db_path = config['db_path'],
        dimensionality = dimensionality,
        lambdas = lambdas,
        mutation_rates = mutation_rates,
        mutation_sizes = mutation_sizes,
        num_evaluation_episodes = num_evaluation_episodes,
        num_training_timesteps = self.num_timesteps,
        ppo_agent = self.model,
        state_type = config['state_type'],
        closeness_to_optimum = config['closeness_to_optimum'],
        episode_seed_generator = main_generator,
      )

      return True

  sb3_seed = main_generator.integers(int(1e9))
  stable_baselines3.common.utils.set_random_seed(sb3_seed)

  # Create a function to instantiate the environment with different seeds
  def create_env():
    seed = main_generator.integers(int(1e9))
    return OneMaxOLL(
      dimensionality = dimensionality,
      seed = seed,
      state_type = config['state_type'],
      action_type = config['action_type'],
      reward_type = config['reward_type'],
      config = config,
    )

  environments = stable_baselines3.common.env_util.make_vec_env(create_env, n_envs=config['num_environments'])

  ppo_agent = stable_baselines3.PPO(
    policy = config['ppo']['policy'],
    env = environments,
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

  evaluate_policy(
    crossover_rates = crossover_rates,
    crossover_sizes = crossover_sizes,
    db_path = config['db_path'],
    dimensionality = dimensionality,
    lambdas = lambdas,
    mutation_rates = mutation_rates,
    mutation_sizes = mutation_sizes,
    num_evaluation_episodes = num_evaluation_episodes,
    num_training_timesteps = 0,
    ppo_agent = ppo_agent,
    state_type = config['state_type'],
    closeness_to_optimum = config['closeness_to_optimum'],
    episode_seed_generator = main_generator,
  )

  ppo_agent.learn(total_timesteps=config['max_training_timesteps'], callback=PPOCallback())

if __name__ == '__main__':
  main()
