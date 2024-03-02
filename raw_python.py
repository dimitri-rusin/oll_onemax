import sqlite3
import inspectify
import onell_algs_rs
import yaml
import unittest
import paper_code.onell_algs

class TestFunctionEvaluations(unittest.TestCase):

  def get_episode_data(self, db_path, episode_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT policy_id, num_function_evaluations, episode_seed FROM episode_info WHERE episode_id = ?', (episode_id,))
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
    cursor.execute('SELECT fitness, lambda_minus_one FROM policies_data WHERE policy_id = ?', (policy_id,))
    rows = cursor.fetchall()
    policy = [lambda_val for fitness, lambda_val in rows]
    conn.close()
    return policy

  def test_evaluation_count(self):
    # Load configuration
    try:
      with open(".env.yaml") as file:
        config = yaml.safe_load(file)
    except FileNotFoundError as e:
      self.fail(f"Error: '.env.yaml' does not exist: {e}")

    episode_id = config['execution']['episode_id']
    db_path = config['db_path']

    # Fetch policy_id, number of function evaluations, and episode_seed from episode_info
    policy_id_to_evaluate, db_num_function_evaluations, episode_seed = self.get_episode_data(db_path, episode_id)
    if policy_id_to_evaluate is None:
      self.fail(f"No policy found for episode_id {episode_id}")

    policy = self.load_policy(db_path, policy_id_to_evaluate)

    for i in range(len(policy)):
      policy[i] += 1

    _, _, python_total_evals = paper_code.onell_algs.onell_lambda(
      n = config['n'],
      seed = episode_seed,
      lbds = policy,
    )

    print("python_total_evals", python_total_evals)

    rust_result = onell_algs_rs.onell_lambda(config['n'], policy, episode_seed, 999_999_999)
    inspectify.d(rust_result)

    self.assertEqual(db_num_function_evaluations, python_total_evals, "Number of function evaluations does not match")

if __name__ == '__main__':
  unittest.main()
