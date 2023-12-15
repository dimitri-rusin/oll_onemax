import gymnasium
import numpy


# Only create one instance of this,
# otherwise the numpy random number generator might get messed up.
class OLL_OneMax(gymnasium.Env):

  def __init__(
    self,
    num_dimensions = 5,
    optimum = None,
    random_seed = None,
  ):
    numpy.random.seed(random_seed)

    self.num_dimensions = num_dimensions
    self.optimum = numpy.random.randint(0, 2, size = self.num_dimensions, dtype = numpy.int32)

    # The assignment is hidden,
    # because we only the previous fitness, the previous lambda, and the current fitness.
    self.assignment = numpy.random.randint(0, 2, size = self.num_dimensions, dtype = numpy.int32)
    self.current_fitness = self.onemax(self.assignment)

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

    info = {}

    return self.current_fitness, info

  def step(self, lamda_minus_one):
    lamda = lamda_minus_one + 1
    previous_fitness = self.current_fitness

    best_assignment_from_first_mutation = None
    best_fitness_from_first_mutation = None
    for _ in range(lamda):
      assignment_from_first_mutation = self.assignment.copy()
      dices = numpy.random.randint(0, self.num_dimensions, size = self.num_dimensions, dtype = numpy.int32)
      assignment_from_first_mutation = numpy.where(dices == 0, 1 - assignment_from_first_mutation, assignment_from_first_mutation)

      new_fitness = self.onemax(assignment_from_first_mutation)
      print(self.assignment)
      print(self.current_fitness)
      print(assignment_from_first_mutation)
      print(new_fitness)
      breakpoint()

      if best_fitness_from_first_mutation is None or new_fitness >= best_fitness_from_first_mutation:
        best_assignment_from_first_mutation = assignment_from_first_mutation
        best_fitness_from_first_mutation = new_fitness

    if self.current_fitness <= best_fitness_from_first_mutation >= best_fitness_from_second_mutation:
      self.current_fitness = best_fitness_from_first_mutation
      self.assignment = best_fitness_from_first_mutation
    elif self.best_fitness_from_second_mutation >= self.current_fitness:
      self.current_fitness = best_fitness_from_second_mutation
      self.assignment = best_assignment_from_second_mutation

    terminated = self.current_fitness == self.num_dimensions
    reward = self.current_fitness - previous_fitness
    info = {}

    return self.current_fitness, reward, terminated, info

  def render(self):
   return self._render_frame()

  def _render_frame(self):
   return "Hello"

  def close(self):
   pass


if __name__ == '__main__':
  oll_onemax = OLL_OneMax()
  print("oll_onemax.optimum")
  print(oll_onemax.optimum)
  print("=========")
  oll_onemax.reset()
  oll_onemax.step(1)
  oll_onemax.close()
