from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy as ep
from stable_baselines3.common.utils import set_random_seed
import collections.abc
import datetime
import gymnasium
import inspectify
import numpy
import onell_algs_rs
import os
import sqlite3
import sys
import time
import torch
import yaml
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import dacbench_adjustments.onell_algs
import ray
from ray.rllib.utils import check_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import Policy



my_restored_policy = Policy.from_checkpoint("computed/ray/2/")
my_restored_policy = my_restored_policy['default_policy']
states = numpy.arange(80).reshape((1, 80))
actions = my_restored_policy.compute_actions(states)
inspectify.d(states)
inspectify.d(actions)
