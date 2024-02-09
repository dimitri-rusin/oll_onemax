import gymnasium
import inspectify
import numpy
import paper_code.onell_algs
import random
import dacbench
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



class OneMaxOll(dacbench.AbstractEnv):
  def __init__(self, config):

    self.n = config['num_dimensions']

    # Required only for AbstractEnv.
    config["action_space"] = gymnasium.spaces.Box(low=1, high=self.n, shape=(1,), dtype=numpy.float32)
    config["benchmark_info"] = "OneMaxOll"
    config["cutoff"] = 9_999
    config["instance_set"] = {'example_instance': None}
    config["observation_space"] = gymnasium.spaces.Box(low=0, high=self.n, shape=(1,), dtype=numpy.float32)
    config["reward_range"] = (-self.n, 0)
    self.config = config

    super(OneMaxOll, self).__init__(self.config)

    self.random_seed = config['random_seed']
    self.observation_space = config["observation_space"]
    self.random_number_generator = numpy.random.default_rng(self.random_seed)

    # WARNING: The optimum for self.x is deterministic: it is the all-ones string.
    self.x = paper_code.onell_algs.OneMax(self.n, rng = self.random_number_generator)

    # We evaluate the onemax function once in the OneMax constructor
    self.agent = None

    self.num_evaluations = 1
    # There is 1 evaluation in the above paper_code.onell_algs.OneMax().

    self.num_policies = 0
    self.num_resets = 0
    self.num_steps = 0
    self.num_steps_across_resets = 0

    with open('ppo_data/episodes.csv', 'w', newline='') as file:
      writer = csv.writer(file, delimiter='|')
      writer.writerow(['Step across episodes', 'Episode', 'Step', 'Fitness', 'Lambda'])

  def set_agent(self, agent):
    self.agent = agent

  def reset_(self):
    self.num_resets += 1
    self.num_steps = 0
    self.random_number_generator = numpy.random.default_rng(self.random_seed)
    self.x = paper_code.onell_algs.OneMax(self.n, rng = self.random_number_generator)
    return numpy.array([self.x.fitness])

  def reset(self, seed = None):
    if seed is not None:
      self.random_seed = seed
    super().reset_()
    return (self.reset_(), "INFO")

  def step(self, lambda_):
    if isinstance(lambda_, numpy.ndarray) and lambda_.size == 1:
      lambda_ = lambda_.item()
    p = lambda_ / self.n
    population_size = numpy.round(lambda_).astype(int)
    prior_fitness = self.x.fitness
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

    if f_y >= self.x.fitness:
      self.x = y

    self.num_evaluations = self.num_evaluations + ne1 + ne2
    self.num_steps += 1
    self.num_steps_across_resets += 1

    num_evaluations_of_this_step = ne1 + ne2
    reward = -num_evaluations_of_this_step + 10 * (self.x.fitness - prior_fitness)
    terminated = self.x.is_optimal()
    truncated = False
    info = {}

    # ======================================================================================

    # SAVE TRAINING EPISODES
    if self.num_steps_across_resets % self.config['episode_steps_omitted_steps'] == 0:
      with open('ppo_data/episodes.csv', 'a', newline = '') as file:
        row = self.num_steps_across_resets, self.num_resets + 1, self.num_steps, prior_fitness, lambda_, reward, self.x.fitness, num_evaluations_of_this_step
        writer = csv.writer(file, delimiter = '|')
        writer.writerow(row)

    # SAVE TRAINED POLICY
    if self.num_steps_across_resets % self.config['policy_omitted_steps'] == 0:
      self.num_policies += 1
      low = int(self.config['observation_space'].low[0])
      high = int(self.config['observation_space'].high[0])
      fitnesses = numpy.arange(low, high).reshape(-1, 1)
      lambdas, _ = self.agent.predict(fitnesses, deterministic = True)
      lambdas_1d = lambdas.flatten().tolist()
      lambdas_1d = [self.num_policies] + lambdas_1d
      with open('ppo_data/policies.csv', 'a', newline = '') as file:
        writer = csv.writer(file, delimiter = '|')
        writer.writerow(lambdas_1d)

    # ======================================================================================

    return numpy.array([self.x.fitness]), reward, terminated, truncated, info
