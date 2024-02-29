import inspectify
import os
import sqlite3
import strategies
import yaml

try:
  with open(".env.yaml") as file:
    global config
    config = yaml.safe_load(file)
except FileNotFoundError:
  print("Error: '.env.yaml' does not exist.", file=sys.stderr)
  sys.exit(1)

n = config['n']
policy_id_to_evaluate = config['execution']['policy_id']
episode_seed = config['execution']['episode_seed']

execution_folder_path = config['execution']['execution_folder_path']

if not os.path.exists(execution_folder_path):
  os.makedirs(execution_folder_path)

# Construct the filename with policy ID and episode seed
filename = f"policy_{policy_id_to_evaluate}_seed_{episode_seed}.txt"

# Combine the folder path and filename
execution_file_path = os.path.join(execution_folder_path, filename)

trace_file = open(execution_file_path, 'w')

def evaluate_policy(env, policy, episode_seed):
  episode_steps = 0
  state = env.reset(episode_seed)[0]  # Pass the episode seed to the reset method
  print("Episode Seed:", episode_seed, file = trace_file)
  done = False
  while not done:
    action = policy[state]
    print("Step:", episode_steps, file = trace_file)
    print("State:", state, file = trace_file)
    print("Action:", action, file = trace_file)
    next_state, reward, done, _ = env.step(action)
    state = next_state[0]
    print("Next State:", state, file = trace_file)
    print("Reward:", reward, file = trace_file)
    episode_steps += 1
    print("------------------------", file = trace_file)
  return episode_steps

def load_env_and_policy(db_path, policy_id):
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  cursor.execute('SELECT fitness, lambda FROM policies_data WHERE policy_id = ?', (policy_id,))
  rows = cursor.fetchall()
  policy = {fitness: lambda_val for fitness, lambda_val in rows}
  env = strategies.OneMaxOLL(n=n)
  conn.close()
  return env, policy

# Evaluate the specified policy for one episode and print , file = trace_filethe episode details
def evaluate_and_print_policy(db_path, policy_id, episode_seed):
  env, policy = load_env_and_policy(db_path, policy_id)
  episode_steps = evaluate_policy(env, policy, episode_seed)
  print("Evaluation Results:", file = trace_file)
  print(f"Episode Steps: {episode_steps}, Episode Seed: {episode_seed}", file = trace_file)

# Assuming the database path is defined
db_path = 'data/policies.db'
evaluate_and_print_policy(db_path, policy_id_to_evaluate, episode_seed)
