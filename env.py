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
      shape = (1,),
      dtype = numpy.float64,
    )

    self.action_space = gymnasium.spaces.Discrete(self.num_dimensions)

  def onemax(self, assignment):
    hamming_distance = numpy.sum(self.optimum != assignment, dtype = numpy.float64)
    normalized_distance = 1 - (hamming_distance / self.num_dimensions)
    return normalized_distance

  def reset(self):
    next_random_seed = numpy.random.randint(0, 10000)
    super().reset(seed = next_random_seed)

    self.optimum = numpy.random.randint(0, 2, size = self.num_dimensions, dtype = numpy.int32)
    self.assignment = numpy.random.randint(0, 2, size = self.num_dimensions, dtype = numpy.int32)
    self.current_fitness = self.onemax(self.assignment)

    self.render()

    info = {}

    return self.current_fitness, info

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

    terminated = self.current_fitness == 1
    reward = self.current_fitness - previous_fitness
    info = {}

    self.render()

    return self.current_fitness, reward, terminated, False, info

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





if __name__ == '__main__':
  with open('trace.csv', 'w') as traces_file:
    print('Sample', '|', 'Fitness', file = traces_file, sep = '')

    env = OLL_OneMax(traces_file, random_seed = 12)
    observation, info = env.reset()
    print(env.optimum)

    for _ in range(1000):
      action = env.action_space.sample()
      observation, reward, terminated, truncated, info = env.step(action)

      if terminated or truncated:
        break

    env.close()
