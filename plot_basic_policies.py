import environment
import gymnasium
import numpy
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
    env = environment.OneMaxOll(config)
    observation = env.reset()
    done = False
    while not done:
      lambda_ = policies[key]['parameters'][env.x.fitness]
      observation, reward, done, info = env.step(lambda_)
      print(f"State: {observation}, Action: {lambda_}, Reward: {reward}, Done: {done}")
    policies[key]['num_iteration_samples'].append(env.num_evaluations)

plotly_figure = plotly.graph_objects.Figure()

for key, values in policies.items():
    plotly_figure.add_trace(plotly.graph_objects.Box(y=values['num_iteration_samples'], name=key))

plotly_figure.update_layout(
    title=f'Distribution of iteration samples for different OLL policies (dimensions: {num_dimensions}, repetitions: {num_repetitions})',
    yaxis_title='Number of iterations',
    xaxis_title='OLL policies'
)

plotly_figure.show()
