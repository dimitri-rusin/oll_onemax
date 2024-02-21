import tuning_environments
import numpy as np
import sqlite3

n = 25
policy_id_to_evaluate = 5
episode_seed = 56088  # Specify the episode seed

trace_path = 'data/single_evaluation.md'
trace_file = open(trace_path, 'w')

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
    cursor.execute('SELECT fitness, lambda FROM policy_log WHERE policy_id = ?', (policy_id,))
    rows = cursor.fetchall()
    policy = {fitness: lambda_val for fitness, lambda_val in rows}
    env = tuning_environments.OneMaxEnv(n=n)
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
