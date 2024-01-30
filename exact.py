
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
    onell_lambda_result = onell_algs.onell_lambda(
      n = num_dimensions,
      lbds = policies[key]['parameters'],
      seed = random_seed,
    )
    policies[key]['num_iteration_samples'].append(onell_lambda_result[2])

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
