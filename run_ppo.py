from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import csv
import inspectify
import numpy
import numpy as np
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

if not os.path.exists('ppo_data/'):
  os.makedirs('ppo_data/')


environment = tuning_environments.OneMaxOll(config)




class TabularSARSAAgent:
  def __init__(self, num_states=10, num_actions=10, alpha=0.1, gamma=0.99, epsilon=0.1):
    self.Q = np.zeros((num_states, num_actions))
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.num_actions = num_actions

  def discretize_action(self, continuous_action):
    """ Discretize the continuous action space. """
    return int((continuous_action - 1) / (10 - 1) * (self.num_actions - 1))

  def choose_action(self, state):
    """ Choose action based on ε-greedy policy. """
    if np.random.rand() < self.epsilon:
      return np.random.choice(self.num_actions)
    else:
      inspectify.d(state)
      inspectify.d(self.Q)
      return np.argmax(self.Q[state])

  def update(self, state, action, reward, next_state, next_action):
    """ Update the Q-table using the SARSA update rule. """
    predict = self.Q[state, action]
    target = reward + self.gamma * self.Q[next_state, next_action]
    self.Q[state, action] += self.alpha * (target - predict)

def train_sarsa(env, agent, num_episodes):
  """ Training loop for the SARSA agent. """
  for episode in range(num_episodes):
    state, _ = env.reset()
    action = agent.choose_action(state)

    while True:
      next_state, reward, done, _ = env.step(agent.discretize_action(action))
      next_action = agent.choose_action(next_state)

      agent.update(state, action, reward, next_state, next_action)

      state = next_state
      action = next_action

      if done:
        break

agent = TabularSARSAAgent()
train_sarsa(environment, agent, num_episodes=1000)
