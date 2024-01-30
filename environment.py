import dacbench
import numpy
import onell_algs

class OneMaxOll(dacbench.AbstractEnv):
  def __init__(self, config):
    super(OneMaxOll, self).__init__(config)
    self.n = config['num_dimensions']

    self.random_number_generator = numpy.random.default_rng(config['random_seed'])
    self.x = onell_algs.OneMax(self.n, rng = self.random_number_generator)

    # We evaluate the onemax function once in the OneMax constructor
    self.num_evaluations = 1
    self.num_steps = 0

  def reset_(self):
    self.state = 0
    return numpy.array([self.state])

  def reset(self):
    super().reset_()
    return self.reset_()

  def step(self, lambda_):
    p = lambda_ / self.n
    population_size = round(lambda_)
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

    reward = self.x.fitness - prior_x.fitness
    done = self.x.is_optimal()
    info = {}

    return numpy.array([self.x.fitness]), reward, done, info
