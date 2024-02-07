import environment
import numpy
import plotly.graph_objects

num_dimensions = 10
num_repetitions = 10
main_random_seed = 42
num_side_random_seeds = 2**32

numpy.random.seed(main_random_seed)
side_random_seeds = numpy.random.randint(num_side_random_seeds, size = num_repetitions)

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
  for side_random_seed in side_random_seeds:
    env = environment.OneMaxOll({
      "num_dimensions": num_dimensions,
      "random_seed": side_random_seed,
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
