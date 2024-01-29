# import gymnasium as gym
# import numpy as np
# from dacbench import AbstractEnv

# class GuessNumberEnv(AbstractEnv):
#     """
#     A simple environment where the agent has to guess a random number.
#     """
#     def __init__(self, config):
#         super(GuessNumberEnv, self).__init__(config)
#         self.target_number = None
#         self.guess_range = self.config["guess_range"]
#         self.observation_space = gym.spaces.Discrete(2)  # Indicates higher or lower
#         self.action_space = gym.spaces.Discrete(self.guess_range)  # Range of guesses

#     def reset_(self):
#         # Randomly select a new target number
#         self.target_number = np.random.randint(1, self.guess_range + 1)
#         return np.array([0])  # Initial observation

#     def reset(self):
#         super().reset_()
#         return self.reset_()

#     def step(self, action):
#         assert self.action_space.contains(action), "Invalid action"
#         if action == self.target_number:
#             reward = 10  # Maximum reward for correct guess
#             done = True
#         else:
#             reward = -abs(action - self.target_number)  # Negative reward based on distance
#             done = False
#         return np.array([1 if action > self.target_number else 0]), reward, done, {}

# # Example usage
# config = {
#     "guess_range": 10,
#     "cutoff": 100,
#     "reward_range": (-9, 10),
#     "observation_space_class": "Discrete",
#     "observation_space_args": [2],
#     "action_space_class": "Discrete",
#     "action_space_args": [10],
#     "seed": 42
# }
# env = GuessNumberEnv(config)
# observation = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()  # Random guess
#     observation, reward, done, info = env.step(action)
#     print(f"Guess: {action}, Observation: {observation}, Reward: {reward}")


# The environment should be like here:
#     /home/dimitri/code/DACBench/dacbench/envs/theory.py
# And here's an example environment that should be similar to this OneMax one:
# /home/dimitri/code/DACBench/dacbench/abstract_env.py




import json
import numpy
import onell_algs
import plotly.graph_objects as go



num_dimensions = 100
num_repetitions = 2

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
  for repetition_index in range(num_repetitions):
    onell_lambda_result = onell_algs.onell_lambda(
      n = num_dimensions,
      lbds = policies[key]['parameters'],
    )
    policies[key]['num_iteration_samples'].append(onell_lambda_result[2])

print(json.dumps(policies, indent = 2))



fig = go.Figure()

for key, values in policies.items():
    fig.add_trace(go.Box(y=values['num_iteration_samples'], name=key))

fig.update_layout(
    title='Distribution of Iteration Samples for Different Lambda Types',
    yaxis_title='Number of Iterations',
    xaxis_title='Lambda Types'
)

fig.show()
