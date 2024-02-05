import environment
import numpy
import plotly.graph_objects

numpy.random.seed(42)

num_dimensions = 100
num_repetitions = 10
random_seeds = numpy.random.randint(2**32, size = num_repetitions)

policies = {
  'theory_lambdas': {
    'parameters': [int(numpy.sqrt(num_dimensions / (num_dimensions - fitness))) for fitness in range(num_dimensions)],
    'num_function_evaluations': [],
  },
  'all_one_lambdas': {
    'parameters': [1] * num_dimensions,
    'num_function_evaluations': [],
  },
  'all_n_lambdas': {
    'parameters': [num_dimensions] * num_dimensions,
    'num_function_evaluations': [],
  },
  'all_n_half_lambdas': {
    'parameters': [num_dimensions // 2] * num_dimensions,
    'num_function_evaluations': [],
  }
}

keys = ['theory_lambdas', 'all_one_lambdas', 'all_n_lambdas', 'all_n_half_lambdas']

for key in keys:
  for random_seed in random_seeds:
    env = environment.OneMaxOll({
      "num_dimensions": num_dimensions,
      "random_seed": random_seed,
    })
    observation = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
      print("env.x.fitness:", env.x.fitness)
      lambda_ = policies[key]['parameters'][env.x.fitness]
      observation, reward, terminated, truncated, info = env.step(lambda_)
      print(f"State: {observation}, Action: {lambda_}, Reward: {reward}, terminated: {terminated}, truncated: {truncated}")
    policies[key]['num_function_evaluations'].append(env.num_evaluations)

plotly_figure = plotly.graph_objects.Figure()

for key, values in policies.items():
    plotly_figure.add_trace(plotly.graph_objects.Box(y=values['num_function_evaluations'], name=key))

plotly_figure.update_layout(
    title=f'Runtime of different OLL policies on OneMax (dimensions: {num_dimensions}, repetitions: {num_repetitions})',
    yaxis_title='Number of function evaluations',
    xaxis_title='OLL policies'
)

plotly_figure.show()
