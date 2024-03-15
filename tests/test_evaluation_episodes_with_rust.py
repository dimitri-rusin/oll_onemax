import sqlite3
import inspectify
import os
import onell_algs_rs
import yaml
import unittest




def load_config_data(db_path):
    config = {}

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM CONFIG")
            rows = cursor.fetchall()

        # Process each row to infer the type and construct a nested dictionary
        config = {}
        for key, value in rows:
            # Infer the type
            if value.isdigit():
                parsed_value = int(value)
            elif all(char.isdigit() or char == '.' for char in value):
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value
            else:
                parsed_value = value

            # Create nested dictionaries based on key structure
            key_parts = key.split('__')
            d = config
            for part in key_parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[key_parts[-1]] = parsed_value
    except sqlite3.Error:
        # If the CONFIG table doesn't exist or any other SQL error occurs
        config = {}

    return config













class TestFunctionEvaluations(unittest.TestCase):

  def get_episode_data(self, db_path, episode_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT policy_id, num_function_evaluations, episode_seed FROM EVALUATION_EPISODES WHERE episode_id = ?', (episode_id,))
    row = cursor.fetchone()
    if not row:
      conn.close()
      return None, None, None
    policy_id, num_function_evaluations, episode_seed = row
    conn.close()
    return policy_id, num_function_evaluations, episode_seed

  def load_policy(self, db_path, policy_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT fitness, lambda_minus_one FROM POLICY_DETAILS WHERE policy_id = ?', (policy_id,))
    rows = cursor.fetchall()
    policy = [lambda_val for fitness, lambda_val in rows]
    conn.close()
    return policy

  def test_evaluation_count(self):
    successful_checks = 0
    failed_checks = 0

    episode_id_low = int(os.getenv('OO__EXECUTION__EPISODE_ID_LOW'))
    episode_id_high = int(os.getenv('OO__EXECUTION__EPISODE_ID_HIGH'))
    db_path = os.getenv('OO__DB_PATH')
    n = int(os.getenv('OO__N'))

    config = load_config_data(db_path)
    probability = config['probability_of_closeness_to_optimum']

    for episode_id in range(episode_id_low, episode_id_high + 1):
      # Fetch policy_id, number of function evaluations, and episode_seed from EVALUATION_EPISODES
      policy_id_to_evaluate, db_num_function_evaluations, episode_seed = self.get_episode_data(db_path, episode_id)
      if policy_id_to_evaluate is None:
        continue

      policy = self.load_policy(db_path, policy_id_to_evaluate)

      for i in range(len(policy)):
        policy[i] += 1

      rust_randomness_num_function_evaluations = onell_algs_rs.onell_lambda(n, policy, episode_seed, 999_999_999, probability)

      try:
        self.assertEqual(db_num_function_evaluations, rust_randomness_num_function_evaluations, "Number of function evaluations does not match for episode_id " + str(episode_id))
        successful_checks += 1
      except AssertionError:
        failed_checks += 1

    print(f"Checked episode IDs from {episode_id_low} to {episode_id_high}.")
    print(f"Successful checks: {successful_checks}")
    print(f"Failed checks: {failed_checks}")

if __name__ == '__main__':
  unittest.main()
