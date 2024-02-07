import gymnasium
import inspectify
import numpy
import paper_code.onell_algs
import random
import dacbench



class OneMaxOll(dacbench.AbstractEnv):
  def __init__(self, config):

    self.n = config['num_dimensions']

    # Required only for AbstractEnv.
    config["action_space"] = gymnasium.spaces.Box(low=1, high=self.n, shape=(1,), dtype=numpy.float32)
    config["benchmark_info"] = "OneMaxOll"
    config["cutoff"] = 9_999
    config["instance_set"] = {'example_instance': None}
    config["observation_space"] = gymnasium.spaces.Box(low=0, high=self.n, shape=(1,), dtype=numpy.float32)
    config["reward_range"] = (-self.n, 0)

    super(OneMaxOll, self).__init__(config)

    self.random_seed = config['random_seed']
    self.observation_space = config["observation_space"]
    self.random_number_generator = numpy.random.default_rng(self.random_seed)

    # WARNING: The optimum for self.x is deterministic: it is the all-ones string.
    self.x = paper_code.onell_algs.OneMax(self.n, rng = self.random_number_generator)

    # We evaluate the onemax function once in the OneMax constructor
    self.num_evaluations = 1
    self.num_steps = 0

  def reset_(self):
    self.random_number_generator = numpy.random.default_rng(self.random_seed)
    self.x = paper_code.onell_algs.OneMax(self.n, rng = self.random_number_generator)
    return numpy.array([self.x.fitness])

  def reset(self, seed = None):
    if seed is not None:
      self.random_seed = seed
    super().reset_()
    return (self.reset_(), "INFO")

  def step(self, lambda_):
    if isinstance(lambda_, numpy.ndarray) and lambda_.size == 1:
      lambda_ = lambda_.item()
    p = lambda_ / self.n
    population_size = numpy.round(lambda_).astype(int)
    xprime, f_xprime, ne1 = self.x.mutate(p, population_size, self.random_number_generator)

    c = 1 / lambda_
    y, f_y, ne2 = self.x.crossover(
      xprime,
      c,
      population_size,
      True,
      True,
      self.random_number_generator,
    )

    prior_x = self.x
    if f_y >= self.x.fitness:
      self.x = y

    self.num_evaluations = self.num_evaluations + ne1 + ne2
    self.num_steps += 1

    reward = -(ne1 + ne2) + (self.x.fitness - prior_x.fitness) + self.x.fitness * lambda_ * 100
    terminated = self.x.is_optimal()
    truncated = False
    info = {}

    return numpy.array([self.x.fitness]), reward, terminated, truncated, info
