import discrete
import inspectify
import numpy as np
import sqlite3



n = 25

def evaluate_policy(env, policy, num_episodes):
    evaluations = []
    for _ in range(num_episodes):
        episode_seed = np.random.randint(100000)
        state = env.reset(episode_seed)[0]  # Pass the episode seed to the reset method
        episode_steps = 0  # Store steps for each episode
        done = False
        while not done:
            action = policy[state]
            next_state, _, done, _ = env.step(action)
            state = next_state[0]
            episode_steps += 1
        evaluations.append((episode_steps, episode_seed))  # Append episode steps and episode seed
    return evaluations  # Return steps for all episodes

def evaluate_all_policies(db_path, env, num_episodes=10):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT policy_id FROM policy_info")
    policy_ids = cursor.fetchall()

    inspectify.d(policy_ids)
    for policy_id in policy_ids:
        inspectify.d(policy_id)
        policy_id = policy_id[0]
        cursor.execute("SELECT fitness FROM policy_log WHERE policy_id=?", (policy_id,))
        policy = cursor.fetchall()
        policy = [int(row[0]) for row in policy]
        episode_steps_list = evaluate_policy(env, policy, num_episodes)

        # Insert steps and episode seed for each episode into the database
        for episode_steps, episode_seed in episode_steps_list:
            cursor.execute("INSERT INTO episode_evaluation (policy_id, steps, episode_seed) VALUES (?, ?, ?)", (policy_id, episode_steps, episode_seed))

    conn.commit()
    conn.close()

# Assuming the database path and environment are defined
db_path = 'ppo_data/policy_data.db'
env = discrete.OneMaxEnv(n=n)
evaluate_all_policies(db_path, env, num_episodes=10)
