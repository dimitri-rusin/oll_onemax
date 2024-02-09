from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import csv
import inspectify
import numpy
import os
import sys
import torch
import tuning_environments
import yaml



try:
  with open(".env.yaml") as file:
    config = yaml.safe_load(file)
except FileNotFoundError:
  print("Error: '.env.yaml' does not exist.", file = sys.stderr)
  os._exit(1)

environment = tuning_environments.OneMaxOll(config)
agent = PPO(
  "MlpPolicy",
  environment,
  verbose=1,
  seed=config['random_seed'],
)

environment.set_agent(agent)

if not os.path.exists('ppo_data/'):
  os.makedirs('ppo_data/')

# SAVE POLICY TRAINING EPISODES, TRAINED POLICY, AND TRAINING ARCHITECTURE
with open("ppo_data/agent_architecture.json", "w") as arch_file:
  print(agent.policy, file = arch_file)
low = int(environment.observation_space.low[0])
high = int(environment.observation_space.high[0])
fitnesses = numpy.arange(low, high).reshape(-1, 1)
fitnesses_1d = fitnesses.flatten().tolist()
fitnesses_1d = [f'Fitness={fitness}' for fitness in fitnesses_1d]
fitnesses_1d = ['ID'] + fitnesses_1d
with open('ppo_data/policies.csv', 'w', newline='') as file:
  writer = csv.writer(file, delimiter='|')
  writer.writerow(fitnesses_1d)

agent.learn(
  total_timesteps = config['total_timesteps'],
)

weights = agent.policy.state_dict()
torch.save(weights, "ppo_data/model_weights.pth")

model_weights_path = 'ppo_data/model_weights.pth'
model_weights = torch.load(model_weights_path)

with open("ppo_data/model_weights.txt", "w") as weights_file:
  for layer_name, weights in model_weights.items():
    print(f"Layer: {layer_name}", file = weights_file)
    print("Weights:", file = weights_file)
    print(weights, file = weights_file)
    print(file = weights_file)
