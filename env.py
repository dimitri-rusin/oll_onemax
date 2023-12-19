from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium
import numpy

class OLL_OneMax(gymnasium.Env):

  def __init__(
    self,
    file = None,
    num_dimensions = 50,
    optimum = None,
    random_seed = None,
  ):
    numpy.random.seed(random_seed)

    self.optimum = None
    self.assignment = None
    self.current_fitness = None
    self.file = file
    self.num_dimensions = num_dimensions

    self.observation_space = gymnasium.spaces.Box(
      low = 0,
      high = 1,
      shape = (self.num_dimensions,),
      dtype = numpy.int32,
    )

    self.action_space = gymnasium.spaces.Discrete(self.num_dimensions * 8)

  def onemax(self, assignment):
    hamming_distance = numpy.sum(self.optimum != assignment, dtype = numpy.float64)
    normalized_distance = 1 - (hamming_distance / self.num_dimensions)
    return normalized_distance

  def reset(self, seed = numpy.random.randint(0, 10000)):
    super().reset(seed = seed)

    self.optimum = numpy.random.randint(0, 2, size = self.num_dimensions, dtype = numpy.int32)
    self.assignment = numpy.zeros(shape = (self.num_dimensions,), dtype = numpy.int32)
    self.current_fitness = self.onemax(self.assignment)

    self.render()

    info = {}

    return self.assignment, info

  def step(self, lamda_minus_one):
    lambda_ = lamda_minus_one + 1
    assert lambda_ >= 1

    previous_fitness = self.current_fitness

    parent = self.assignment
    for _ in range(2):
      best_offspring_assignment = None
      best_offspring_fitness = None
      for _ in range(lambda_):
        current_offspring_assignment = parent.copy()
        mutation_chance = numpy.random.rand(self.num_dimensions) < 1 / self.num_dimensions
        current_offspring_assignment = numpy.where(
          mutation_chance,
          1 - current_offspring_assignment,
          current_offspring_assignment,
        )
        current_offspring_fitness = self.onemax(current_offspring_assignment)
        if best_offspring_fitness is None or current_offspring_fitness >= best_offspring_fitness:
          best_offspring_assignment = current_offspring_assignment
          best_offspring_fitness = current_offspring_fitness

      parent = best_offspring_assignment
      if best_offspring_fitness >= self.current_fitness:
        self.assignment = best_offspring_assignment
        self.current_fitness = best_offspring_fitness

    terminated = bool(self.current_fitness == 1)
    reward = -1
    info = {}

    self.render()

    return self.assignment, reward, terminated, False, info

  def render(self):

    print(
      ''.join(str(bit) for bit in self.assignment),
      '|',
      self.current_fitness,
      file = self.file,
      sep = ''
    )

  def close(self):
   pass




def evaluate_model(env, model, num_episodes=50):
  total_reward = 0
  for episode in range(num_episodes):
    observation, _ = env.reset()
    terminated = False
    while not terminated:
      action, _ = model.predict(observation, deterministic = True)
      observation, reward, terminated, _, _ = env.step(action)
      total_reward += reward
  average_reward = total_reward / num_episodes
  return average_reward



if __name__ == '__main__':
  with open('trace.csv', 'w') as traces_file:
    print('Sample', '|', 'Fitness', file = traces_file, sep = '')

    env = OLL_OneMax(traces_file, random_seed = 12)

    check_env(env)

    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose = 1)

    average_reward = evaluate_model(env, model, num_episodes = 50)
    print(f"Expected average reward across 50 episodes: {average_reward}")

    for obs in [i * 0.1 for i in range(10)]:
      action, _ = model.predict(numpy.random.randint(0, 2, size = env.num_dimensions, dtype = numpy.int32), deterministic = True)
      print(f"Map {obs} -> {action}")

    # Train the agent
    model.learn(total_timesteps = 10_000)

    average_reward = evaluate_model(env, model, num_episodes = 50)
    print(f"Expected average reward across 50 episodes: {average_reward}")

    for obs in [i * 0.1 for i in range(10)]:
      action, _ = model.predict(numpy.random.randint(0, 2, size = env.num_dimensions, dtype = numpy.int32), deterministic = True)
      print(f"Map {obs} -> {action}")

    observation, info = env.reset()
    print(env.optimum)
    for _ in range(1_000):
      action, _ = model.predict(observation, deterministic = True)
      observation, reward, terminated, truncated, info = env.step(action)

      if terminated or truncated:
        break

    env.close()
