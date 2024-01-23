import gymnasium
import numpy

class OLL_OneMax(gymnasium.Env):

  def __init__(
    self,
    file,
    num_dimensions,
  ):

    self.lambdas = []
    self.assignments = []
    self.fitnesses = []

    self.assignment = None
    self.current_fitness = None
    self.file = file
    self.num_dimensions = num_dimensions
    self.optimum = None

    self.observation_space = gymnasium.spaces.Box(
      low = 0,
      high = 1,
      shape = (1,),
      dtype = numpy.int32,
    )

    self.action_space = gymnasium.spaces.Discrete(self.num_dimensions * 8)

  def onemax(self, assignment):
    hamming_distance = numpy.sum(self.optimum != assignment, dtype = numpy.float64)
    relative_similarity = 1 - (hamming_distance / self.num_dimensions)
    return relative_similarity

  def reset(self, seed = numpy.random.randint(0, 10_000)):
    super().reset(seed = seed)

    self.optimum = numpy.random.randint(0, 2, size = self.num_dimensions, dtype = numpy.int32)
    self.assignment = numpy.zeros(shape = (self.num_dimensions,), dtype = numpy.int32)
    self.current_fitness = self.onemax(self.assignment)
    info = {}

    return self.assignment, info

  def step(self, lambda_minus_one):
    lambda_ = lambda_minus_one + 1
    assert lambda_ >= 1

    self.assignments.append(self.assignment)
    self.fitnesses.append(self.current_fitness)
    self.lambdas.append(lambda_)

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
    reward = -1 * 2 * lambda_
    info = {}
    if terminated:
      self.render()

    return self.current_fitness, reward, terminated, False, info

  # Just log the behaviour to a file.
  def render(self):

    for index, (assignment, fitness, lambda_) in enumerate(zip(self.assignments, self.fitnesses, self.lambdas)):
      num_function_evaluations = sum(self.lambdas[index:]) * 2
      print(
        ''.join(str(bit) for bit in assignment),
        '|',
        fitness,
        '|',
        lambda_,
        '|',
        num_function_evaluations,
        file = self.file,
        sep = ''
      )

  def close(self):
   pass







def evaluate_random_behaviour(environment, num_episodes = 50):
  total_reward = 0
  for episode in range(num_episodes):
    observation, _ = environment.reset()
    terminated = False
    while not terminated:
      theoretical_lambda = numpy.sqrt(1. / (1. - environment.current_fitness))
      theoretical_lambda = round(theoretical_lambda)
      observation, reward, terminated, _, _ = environment.step(theoretical_lambda - 1)
      total_reward += reward
  average_reward = total_reward / num_episodes
  return average_reward

if __name__ == '__main__':
  random_seed = 12
  numpy.random.seed(random_seed)

  with open('traces.csv', 'w') as traces_file:
    print('Sample', '|', 'Fitness', '|', 'Lambda', '|', 'Budget', file = traces_file, sep = '')
    environment = OLL_OneMax(
      file = traces_file,
      num_dimensions = 1000
    )
    average_reward = evaluate_random_behaviour(environment)
    print(f"Expected average reward across 50 episodes: {average_reward}")
    environment.close()
