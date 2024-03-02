import sqlite3
import onell_algs_rs
import yaml
import unittest

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

    # Load configuration
    try:
      with open(".env.yaml") as file:
        config = yaml.safe_load(file)
    except FileNotFoundError as e:
      self.fail(f"Error: '.env.yaml' does not exist: {e}")

    episode_id_low = config['execution']['episode_id_low']
    episode_id_high = config['execution']['episode_id_high']
    db_path = config['db_path']

    for episode_id in range(episode_id_low, episode_id_high + 1):
      # Fetch policy_id, number of function evaluations, and episode_seed from EVALUATION_EPISODES
      policy_id_to_evaluate, db_num_function_evaluations, episode_seed = self.get_episode_data(db_path, episode_id)
      if policy_id_to_evaluate is None:
        self.fail(f"No policy found for episode_id {episode_id}")

      policy = self.load_policy(db_path, policy_id_to_evaluate)

      for i in range(len(policy)):
        policy[i] += 1

      rust_randomness_num_function_evaluations = onell_algs_rs.onell_lambda(config['n'], policy, episode_seed, 999_999_999)

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
