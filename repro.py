import discrete
import numpy as np
import sqlite3

n = 25
policy_id_to_evaluate = 1
episode_seed = 12345  # Specify the episode seed

def evaluate_policy(env, policy, episode_seed):
    episode_steps = 0
    state = env.reset(episode_seed)[0]  # Pass the episode seed to the reset method
    print("Episode Seed:", episode_seed)
    done = False
    while not done:
        action = policy[state]
        print("Step:", episode_steps)
        print("State:", state)
        print("Action:", action)
        next_state, reward, done, _ = env.step(action)
        state = next_state[0]
        print("Next State:", state)
        print("Reward:", reward)
        episode_steps += 1
        print("------------------------")
    return episode_steps

# Load the environment and policy from the database
def load_env_and_policy(db_path, policy_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT fitness FROM policy_log WHERE policy_id=?", (policy_id,))
    policy = cursor.fetchone()[0]
    env = discrete.OneMaxEnv(n=n)
    conn.close()
    return env, policy

# Evaluate the specified policy for one episode and print the episode details
def evaluate_and_print_policy(db_path, policy_id, episode_seed):
    env, policy = load_env_and_policy(db_path, policy_id)
    episode_steps = evaluate_policy(env, policy, episode_seed)
    print("Evaluation Results:")
    print(f"Episode Steps: {episode_steps}, Episode Seed: {episode_seed}")

# Assuming the database path is defined
db_path = 'ppo_data/policy_data.db'
evaluate_and_print_policy(db_path, policy_id_to_evaluate, episode_seed)
