import sqlite3
import sys
import yaml

def query_table_summary(db_path, table_name, stats_columns, policy_id_analysis=False):
  """Query and print statistics for the given table and columns."""
  with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()

    # Get the count of rows in the table
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cursor.fetchone()[0]
    print(f"\nTable '{table_name}' has {row_count} rows.")

    # Get statistics for specified columns
    for column in stats_columns:
      cursor.execute(f"SELECT AVG({column}), MIN({column}), MAX({column}) FROM {table_name};")
      avg, min_val, max_val = cursor.fetchone()

      # Calculate standard deviation
      cursor.execute(f"SELECT AVG(({column} - ?) * ({column} - ?)) FROM {table_name};", (avg, avg))
      stddev = (cursor.fetchone()[0] ** 0.5) if row_count > 1 else 0

      print(f"Column '{column}': AVG = {avg}, STDDEV = {stddev:.2f}, MIN = {min_val}, MAX = {max_val}")

    # Policy ID Analysis
    if policy_id_analysis:
      analyze_policy_id(conn, table_name)

def analyze_policy_id(conn, table_name):
  """Analyze the policy_id column for missing integers."""
  cursor = conn.cursor()
  cursor.execute(f"SELECT MIN(policy_id), MAX(policy_id) FROM {table_name};")
  min_id, max_id = cursor.fetchone()
  print(f"Column 'policy_id': MIN = {min_id}, MAX = {max_id}")

  cursor.execute(f"SELECT policy_id FROM {table_name} ORDER BY policy_id;")
  all_ids = [row[0] for row in cursor.fetchall()]
  missing_count = len(set(range(min_id, max_id + 1)) - set(all_ids))
  print(f"Number of missing integers in 'policy_id' between {min_id} and {max_id}: {missing_count}")

def main():
  try:
    with open(".env.yaml") as file:
      config = yaml.safe_load(file)
  except FileNotFoundError:
    print("Error: '.env.yaml' does not exist.", file=sys.stderr)
    sys.exit(1)

  db_path = config['db_path']

  # Define tables and their relevant columns for statistics
  tables_stats = {
    'policies_data': ['fitness', 'lambda'],
    'episode_lengths': ['episode_length'],
    'policies_info': []  # No additional stats columns, but will do policy_id analysis
  }

  for table, columns in tables_stats.items():
    query_table_summary(db_path, table, columns, policy_id_analysis=True)

if __name__ == "__main__":
  main()
