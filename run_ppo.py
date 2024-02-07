from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import BaseCallback
import csv
import inspectify
import numpy
import os
import tuning_environments



class PolicyLoggerCallback(BaseCallback):
  def __init__(self, policy_omitted_steps, episode_steps_omitted_steps, filename):
    super(PolicyLoggerCallback, self).__init__()
    self.counter = 0
    self.episode_steps_omitted_steps = episode_steps_omitted_steps
    self.filename = filename
    self.num_policies = 0
    self.policy_omitted_steps = policy_omitted_steps

  def _on_step(self) -> bool:
    self.counter += 1

    # SAVE TRAINING EPISODES
    if self.counter % self.episode_steps_omitted_steps == 0:
      with open('ppo_data/episodes.csv', 'a', newline='') as file:
        already_arrived = any(self.locals['dones'])
        num_episodes = len(self.locals['env'].envs[0].episode_lengths) + 1
        total_steps = self.locals['env'].envs[0].total_steps
        num_steps = total_steps - sum(self.locals['env'].envs[0].episode_lengths)
        if already_arrived:
          num_episodes -= 1
          num_steps += self.locals['env'].envs[0].episode_lengths[-1]
        fitness = self.locals['obs_tensor'].item()
        lambda_ = self.locals['clipped_actions'].item()
        row = total_steps,num_episodes, num_steps, fitness, lambda_
        writer = csv.writer(file, delimiter='|')
        writer.writerow(row)

    # SAVE TRAINED POLICY
    if self.counter % self.policy_omitted_steps == 0:
      self.num_policies += 1
      low = int(self.locals['env'].envs[0].env.observation_space.low[0])
      high = int(self.locals['env'].envs[0].env.observation_space.high[0])
      fitnesses = numpy.arange(low, high).reshape(-1, 1)
      lambdas, _ = self.model.predict(fitnesses, deterministic=True)
      lambdas_1d = lambdas.flatten().tolist()
      lambdas_1d = [self.num_policies] + lambdas_1d
      with open('ppo_data/policies.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerow(lambdas_1d)

    return True



config = {
  'num_dimensions': 10,
  'random_seed': 12345,
}
environment = tuning_environments.OneMaxOll(config)
model = PPO(
  "MlpPolicy",
  environment,
  learning_rate=0.0003,
  policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
  verbose=1,
  seed=config['random_seed'],
)

if not os.path.exists('ppo_data/'):
  os.makedirs('ppo_data/')

# SAVE POLICY TRAINING EPISODES, TRAINED POLICY, AND TRAINING ARCHITECTURE
with open("ppo_data/agent_architecture.json", "w") as arch_file:
  print(model.policy, file = arch_file)
low = int(environment.observation_space.low[0])
high = int(environment.observation_space.high[0])
fitnesses = numpy.arange(low, high).reshape(-1, 1)
fitnesses_1d = fitnesses.flatten().tolist()
fitnesses_1d = [f'Fitness={fitness}' for fitness in fitnesses_1d]
fitnesses_1d = ['ID'] + fitnesses_1d
with open('ppo_data/policies.csv', 'w', newline='') as file:
  writer = csv.writer(file, delimiter='|')
  writer.writerow(fitnesses_1d)
with open('ppo_data/episodes.csv', 'w', newline='') as file:
  writer = csv.writer(file, delimiter='|')
  writer.writerow(['Step across episodes', 'Episode', 'Step','Fitness','Lambda'])



policy_logger_callback = PolicyLoggerCallback(
  episode_steps_omitted_steps=100,
  policy_omitted_steps=5_000,
  filename="policies.csv",
)

model.learn(total_timesteps=20_000, callback=[policy_logger_callback])
