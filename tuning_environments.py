import gymnasium
import numpy as np
import os
import plotly.graph_objects as go
import sqlite3



class OneMaxEnv(gymnasium.Env):
  def __init__(self, n, seed=None):
    super(OneMaxEnv, self).__init__()
    self.n = n
    self.action_space = gymnasium.spaces.Discrete(n)
    self.observation_space = gymnasium.spaces.Box(low=0, high=n - 1, shape=(1,), dtype=np.int32)
    self.seed = seed
    self.random = np.random.RandomState(self.seed)
    assert seed is None
    self.optimum = None
    self.current_solution = None
    self.evaluations = {}

  def reset(self, episode_seed):
    # Use the provided seed for reproducibility
    self.seed = episode_seed
    self.random = np.random.RandomState(self.seed)

    self.current_solution = np.zeros(self.n, dtype=int)
    self.optimum = self.random.randint(2, size=self.n)

    # Set approximately 85% of the bits to 1
    num_ones = int(self.n * 0.5)
    one_positions = self.random.choice(self.n, num_ones, replace=False)
    self.current_solution[one_positions] = 1

    self.evaluations = {}
    return np.array([self.evaluate(self.current_solution)])

  def step(self, action):
    λ = action + 1
    offspring = self.generate_offspring(λ)
    self.current_solution, evaluations_this_step = self.select_solution(offspring)
    fitness = self.evaluate(self.current_solution)
    reward = -evaluations_this_step
    done = fitness == self.n
    return np.array([fitness]), reward, done, {}

  def evaluate(self, solution):
    solution_key = tuple(solution)
    if solution_key in self.evaluations:
      return self.evaluations[solution_key]
    fitness = np.sum(solution)
    self.evaluations[solution_key] = fitness
    return fitness

  def generate_offspring(self, λ):
    offspring = []
    for _ in range(λ):
      mutated = self.mutate(self.current_solution)
      crossed = self.crossover(self.current_solution, mutated)
      offspring.append(crossed)
    return offspring

  def mutate(self, solution):
    mutation = self.random.randint(2, size=self.n)
    return np.bitwise_xor(solution, mutation)

  def crossover(self, parent, other):
    mask = self.random.randint(2, size=self.n)
    return np.where(mask, parent, other)

  def select_solution(self, offspring):
    evaluations_this_step = 0
    best_solution = self.current_solution
    best_fitness = self.evaluate(self.current_solution)
    for child in offspring:
      fitness = self.evaluate(child)
      if fitness > best_fitness:
        best_solution = child
        best_fitness = fitness
      evaluations_this_step += 1
    return best_solution, evaluations_this_step
