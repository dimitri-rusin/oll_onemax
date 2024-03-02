import gymnasium
import strategies
import inspectify
import numpy
import os
import paper_code.onell_algs
import sqlite3
import sys
import time
import yaml



def load_policy(db_path, policy_id):
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  cursor.execute('SELECT fitness, lambda_minus_one FROM policies_data WHERE policy_id = ?', (policy_id,))
  rows = cursor.fetchall()
  policy = [lambda_val for fitness, lambda_val in rows]
  conn.close()
  return policy

try:
  with open(".env.yaml") as file:
    global config
    config = yaml.safe_load(file)
except FileNotFoundError:
  print("Error: '.env.yaml' does not exist.", file=sys.stderr)
  sys.exit(1)

policy_id_to_evaluate = config['execution']['policy_id']
episode_seed = config['execution']['episode_seed']

policy = load_policy(config['db_path'], policy_id_to_evaluate)

for i in range(len(policy)):
  policy[i] += 1

x, f_x, total_evals = paper_code.onell_algs.onell_lambda(
  n = 50,
  seed = episode_seed,
  lbds = policy,
)

inspectify.d(total_evals)
