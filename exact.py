import dacbench
import gymnasium
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




import json
import numpy
import onell_algs
import plotly.graph_objects

numpy.random.seed(42)

num_dimensions = 20
num_repetitions = 20
random_seeds = numpy.random.randint(2**32, size = num_repetitions)

policies = {
  'theory_lambdas': {
    'parameters': [int(numpy.sqrt(num_dimensions / (num_dimensions - fitness))) for fitness in range(num_dimensions)],
    'num_iteration_samples': [],
  },
  'all_one_lambdas': {
    'parameters': [1] * num_dimensions,
    'num_iteration_samples': [],
  },
  'all_n_lambdas': {
    'parameters': [num_dimensions] * num_dimensions,
    'num_iteration_samples': [],
  },
  'all_n_half_lambdas': {
    'parameters': [num_dimensions // 2] * num_dimensions,
    'num_iteration_samples': [],
  }
}

keys = ['theory_lambdas', 'all_one_lambdas', 'all_n_lambdas', 'all_n_half_lambdas']

for key in keys:
  for random_seed in random_seeds:

    config = {
      "random_seed": random_seed,
      "num_dimensions": num_dimensions,

      # not used, required for AbstractEnv
      "action_space": gymnasium.spaces.Discrete(10),
      "benchmark_info": "Simple Increment Environment",
      "cutoff": 100,
      "instance_set": {'example_instance': None},
      "observation_space": gymnasium.spaces.Discrete(100),
      "reward_range": (0, 1),
    }
    env = OneMaxOll(config)
    observation = env.reset()
    done = False
    while not done:
      lambda_ = policies[key]['parameters'][env.x.fitness]
      observation, reward, done, info = env.step(lambda_)
      print(f"State: {observation}, Action: {lambda_}, Reward: {reward}, Done: {done}")
    policies[key]['num_iteration_samples'].append(env.num_evaluations)

print(json.dumps(policies, indent = 2))



plotly_figure = plotly.graph_objects.Figure()

for key, values in policies.items():
    plotly_figure.add_trace(plotly.graph_objects.Box(y=values['num_iteration_samples'], name=key))

plotly_figure.update_layout(
    title=f'Distribution of iteration samples for different OLL policies (dimensions: {num_dimensions}, repetitions: {num_repetitions})',
    yaxis_title='Number of iterations',
    xaxis_title='OLL policies'
)

plotly_figure.show()
