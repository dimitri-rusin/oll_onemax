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

config = None

# ============== ENVIRONMENT - BEGIN ==============

class OneMaxOLL(gymnasium.Env):
  def __init__(self, dimensionality, seed, reward_type, action_type, state_type):
    super(OneMaxOLL, self).__init__()
    self.dimensionality = dimensionality

    # We use a numpy array, because λ is a numpy.int64 in step().
    self.mutation_sizes = numpy.array(config['mutation_sizes'], dtype = numpy.int64)
    self.crossover_sizes = numpy.array(config['crossover_sizes'], dtype = numpy.int64)

    assert action_type in ['DISCRETE', 'CONTINUOUS']
    if action_type == 'DISCRETE':
      self.action_space = gymnasium.spaces.Tuple((
        gymnasium.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=numpy.float64),
        gymnasium.spaces.Discrete(n=self.mutation_sizes.shape[0]),
        gymnasium.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=numpy.float64),
        gymnasium.spaces.Discrete(n=self.crossover_sizes.shape[0]),
      ))
    if action_type == 'CONTINUOUS':
      assert False, "Check Continuous action space logic in step()."
      self.action_space = gymnasium.spaces.Box(
        low=config['continuous_actions_high'],
        high=config['continuous_actions_high'],
        shape=(1,),
        dtype=numpy.float64,
      )

    if state_type == 'SCALAR_ENCODED':
      self.observation_space = gymnasium.spaces.Box(low=0, high=dimensionality - 1, shape=(1,), dtype=numpy.int32)
    if state_type == 'ONE_HOT_ENCODED':
      self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(dimensionality,), dtype=numpy.int32)
    self.state_type = state_type

    self.seed = seed
    self.random_number_generator = numpy.random.default_rng(self.seed)
    self.current_solution = None
    self.num_function_evaluations = None
    self.num_total_timesteps = 0
    self.num_total_function_evaluations = 0
    self.reward_type = reward_type
    self.num_timesteps_with_teacher = config['num_timesteps_with_teacher']

  def reset(self, seed = None):
    if seed is not None:
      self.seed = seed
    self.num_function_evaluations = 0
    self.random_number_generator = numpy.random.default_rng(self.seed)

    initial_fitness = self.dimensionality
    while initial_fitness >= self.dimensionality:
      self.current_solution = DE0CH_OLL.tuned_with_irace.onell_algs.OneMax(
        self.dimensionality,
        rng = self.random_number_generator,
        ratio_of_optimal_bits = config['probability_of_closeness_to_optimum'],
      )
      initial_fitness = self.current_solution.fitness

    # there is one evaluation [call to .eval()] inside OneMax
    self.num_function_evaluations += 1

    fitness_array = None
    if self.state_type == 'SCALAR_ENCODED':
      fitness_array = numpy.array([self.current_solution.fitness])
    if self.state_type == 'ONE_HOT_ENCODED':
      fitness_array = create_fitness_vector(self.current_solution.fitness)

    return fitness_array, {}

  def step(self, action_index):

    λ = self.actions[action_index]
    prior_fitness = self.current_solution.fitness

    population_size = numpy.round(λ).astype(int)
    p = λ / self.dimensionality
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

    if self.num_total_timesteps < self.num_timesteps_with_teacher:
      reward = -(λ - int(numpy.sqrt(self.dimensionality / (self.dimensionality - prior_fitness)))) ** 2
    else:
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

def evaluate_episode(lambda_policy, episode_seed):
  """Simulate an episode based on the lambda_policy and return fitness at each step."""

  policy_list = [λ for _, λ in lambda_policy.items()]

  dimensionality = len(lambda_policy)
  num_function_evaluations, num_evaluation_timesteps = onell_algs_rs.onell_lambda(
    dimensionality,
    policy_list,
    episode_seed,
    999_999_999,
    config['probability_of_closeness_to_optimum'],
  )

  return num_function_evaluations, num_evaluation_timesteps

def create_fitness_vector(fitness):
  fitness_vector = numpy.zeros(config['dimensionality'], dtype=numpy.int32)
  if fitness < config['dimensionality']:
    fitness_vector[fitness] = 1
  return fitness_vector

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

def flatten_config(prefix, nested_config):
  items = []
  for key, value in nested_config.items():
    new_key = f"{prefix}__{key}" if prefix else key
    if isinstance(value, dict):
      items.extend(flatten_config(new_key, value))
    else:
      items.append((new_key, value))
  return items



def main():

  load_config()

  numpy.random.seed(config['random_seed'])

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

      CREATE TABLE IF NOT EXISTS EVALUATION_EPISODES (policy_id INTEGER, episode_id INTEGER PRIMARY KEY AUTOINCREMENT, episode_seed INTEGER, num_timesteps INTEGER, num_function_evaluations INTEGER, FOREIGN KEY(policy_id) REFERENCES POLICY_DETAILS(policy_id));
      CREATE TABLE IF NOT EXISTS CONFIG (key TEXT PRIMARY KEY, value TEXT);
    ''')

  flattened_config = flatten_config('', config)

  with database:
    for key, value in flattened_config:
      database.execute('INSERT INTO CONFIG (key, value) VALUES (?, ?)', (key, str(value)))

  dimensionality = config['dimensionality']
  policy_id = -1
  theory_derived_lambda_policy = {fitness : int(numpy.sqrt(dimensionality / (dimensionality - fitness))) for fitness in range(dimensionality)}
  theory_derived_lambda_policy = {int(fitness): (lambda_ / dimensionality, lambda_, 1. / lambda_, lambda_) for fitness, lambda_ in theory_derived_lambda_policy.items()}
  with database:
    cursor = database.cursor()
    if policy_id is None:
      cursor.execute('INSERT INTO CONSTRUCTED_POLICIES (num_training_episodes) VALUES (?);', (0,))
      policy_id = cursor.lastrowid
    else:
      cursor.execute(
        'INSERT INTO CONSTRUCTED_POLICIES '
        '(policy_id, num_total_timesteps, num_training_episodes, '
        'num_total_function_evaluations, mean_initial_fitness, variance_initial_fitness) '
        'VALUES (?, ?, ?, ?, ?, ?);',
        (policy_id, 0, 0, 0, 0, 0)
      )
      cursor.executemany(
        'INSERT INTO POLICY_DETAILS (policy_id, fitness, mutation_rate, mutation_size, crossover_rate, crossover_size) VALUES (?, ?, ?, ?, ?, ?);',
        [(policy_id, fitness, mutation_rate, mutation_size, crossover_rate, crossover_size) for fitness, (mutation_rate, mutation_size, crossover_rate, crossover_size) in theory_derived_lambda_policy.items()]
      )

  db_path = config['db_path']
  num_evaluation_episodes = config['num_evaluation_episodes']


  seed_for_generating_episode_seeds = numpy.random.randint(999_999)
  episode_seed_generator = numpy.random.RandomState(seed_for_generating_episode_seeds)
  episode_data = []
  for episode_index in range(num_evaluation_episodes):
    print(f"Policy {policy_id}: Evaluating episode {episode_index + 1} / {num_evaluation_episodes}.")
    episode_seed = episode_seed_generator.randint(999_999)
    num_function_evaluations, num_evaluation_timesteps = evaluate_episode(theory_derived_lambda_policy, episode_seed)
    episode_data.append((policy_id, episode_seed, num_evaluation_timesteps, num_function_evaluations))

  with sqlite3.connect(db_path, timeout=10) as database:
    with database:
      cursor = database.cursor()
      cursor.executemany(
        'INSERT INTO EVALUATION_EPISODES (policy_id, episode_seed, num_timesteps, num_function_evaluations) VALUES (?, ?, ?, ?);',
        episode_data
      )

  lambdas = numpy.array(config['lambdas'], dtype = numpy.int64)

  class PPOCallback(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self, verbose=0):
      super(PPOCallback, self).__init__(verbose)
      self.evaluation_results = []

    def _on_step(self):

      # self.num_timesteps is the number of timesteps made across all environments, for each environment we have made n_steps steps.
      if self.num_timesteps % config['num_timesteps_per_evaluation'] == 0:

        if config['state_type'] == 'ONE_HOT_ENCODED':
          index_policy = {obs_value: self.model.predict(create_fitness_vector(obs_value), deterministic=True)[0].item() for obs_value in range(dimensionality)}
        if config['state_type'] == 'SCALAR_ENCODED':
          index_policy = {obs_value: self.model.predict(numpy.array([obs_value]).reshape((1, 1)), deterministic=True)[0][0] for obs_value in range(dimensionality)}

        lambda_policy = {}
        for observation, action_index in index_policy.items():
          lambda_policy[observation] = lambdas[action_index]

        with sqlite3.connect(config['db_path']) as database:
          cursor = database.cursor()
          current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
          cursor.execute('INSERT INTO CONSTRUCTED_POLICIES (num_total_timesteps, created_at) VALUES (?, ?);', (self.num_timesteps, current_time))
          policy_id = cursor.lastrowid
          policy_data = [(int(policy_id), int(fitness), int(lambda_)) for fitness, lambda_ in lambda_policy.items()]
          cursor.executemany('INSERT INTO POLICY_DETAILS (policy_id, fitness, lambda) VALUES (?, ?, ?);', policy_data)

        seed_for_generating_episode_seeds = numpy.random.randint(999_999)
        episode_seed_generator = numpy.random.RandomState(seed_for_generating_episode_seeds)
        num_function_evaluations_list = []

        for episode_index in range(num_evaluation_episodes):
          print(f"Policy: Evaluating episode {episode_index + 1} / {num_evaluation_episodes}.")
          episode_seed = episode_seed_generator.randint(999_999)
          num_function_evaluations, num_evaluation_timesteps = evaluate_episode(lambda_policy, episode_seed)
          num_function_evaluations_list.append((policy_id, episode_seed, num_evaluation_timesteps, num_function_evaluations))

        with sqlite3.connect(config['db_path'], timeout=10) as database:
          cursor = database.cursor()
          cursor.executemany(
            'INSERT INTO EVALUATION_EPISODES (policy_id, episode_seed, num_timesteps, num_function_evaluations) VALUES (?, ?, ?, ?);',
            num_function_evaluations_list
          )

        only_fitness = [int(lambda_) for fitness, lambda_ in lambda_policy.items()]
        print("lambda_policy", only_fitness)
        only_fitness = [int(lambda_) for fitness, lambda_ in index_policy.items()]
        print("index_policy", only_fitness)

      return True

  sb3_seed = numpy.random.randint(999_999)
  stable_baselines3.common.utils.set_random_seed(sb3_seed)

  # Create a function to instantiate the environment with different seeds
  def create_env():
    seed = numpy.random.randint(999_999)
    return OneMaxOLL(
      dimensionality=dimensionality,
      seed=seed,
      actions=config['lambdas'],
      state_type=config['state_type'],
      action_type=config['action_type'],
      reward_type=config['reward_type']
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

  # ================== EVALUATION OF FIRST POLICY ==================================================
  if config['state_type'] == 'ONE_HOT_ENCODED':
    index_policy = {obs_value: ppo_agent.predict(create_fitness_vector(obs_value), deterministic=True)[0].item() for obs_value in range(dimensionality)}
  if config['state_type'] == 'SCALAR_ENCODED':
    index_policy = {obs_value: ppo_agent.predict(numpy.array([obs_value]).reshape((1, 1)), deterministic=True)[0][0] for obs_value in range(dimensionality)}

  lambda_policy = {}
  for observation, action_index in index_policy.items():
    lambda_policy[observation] = lambdas[action_index]

  with sqlite3.connect(config['db_path']) as database:
    cursor = database.cursor()
    current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cursor.execute('INSERT INTO CONSTRUCTED_POLICIES (num_total_timesteps, created_at) VALUES (?, ?);', (0, current_time))
    policy_id = cursor.lastrowid
    policy_data = [(int(policy_id), int(fitness), int(lambda_)) for fitness, lambda_ in lambda_policy.items()]
    cursor.executemany('INSERT INTO POLICY_DETAILS (policy_id, fitness, lambda) VALUES (?, ?, ?);', policy_data)

  seed_for_generating_episode_seeds = numpy.random.randint(999_999)
  episode_seed_generator = numpy.random.RandomState(seed_for_generating_episode_seeds)
  num_function_evaluations_list = []

  for episode_index in range(num_evaluation_episodes):
    print(f"Policy: Evaluating episode {episode_index + 1} / {num_evaluation_episodes}.")
    episode_seed = episode_seed_generator.randint(999_999)
    num_function_evaluations, num_evaluation_timesteps = evaluate_episode(lambda_policy, episode_seed)
    num_function_evaluations_list.append((policy_id, episode_seed, num_evaluation_timesteps, num_function_evaluations))

  with sqlite3.connect(config['db_path'], timeout=10) as database:
    cursor = database.cursor()
    cursor.executemany(
      'INSERT INTO EVALUATION_EPISODES (policy_id, episode_seed, num_timesteps, num_function_evaluations) VALUES (?, ?, ?, ?);',
      num_function_evaluations_list
    )

  only_fitness = [int(lambda_) for fitness, lambda_ in lambda_policy.items()]
  print("lambda_policy", only_fitness)
  only_fitness = [int(lambda_) for fitness, lambda_ in index_policy.items()]
  print("index_policy", only_fitness)
  # ================== EVALUATION OF FIRST POLICY ==================================================



  ppo_agent.learn(total_timesteps=config['max_training_timesteps'], callback=PPOCallback())



if __name__ == '__main__':
  main()
