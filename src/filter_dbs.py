import sqlite3
import argparse
import os
from collections import defaultdict

def check_keys(db_path, keys):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        key_values = []
        for key in keys:
            cursor.execute("SELECT value FROM CONFIG WHERE key=?", (key,))
            row = cursor.fetchone()
            if row:
                key_values.append(row[0])
            else:
                key_values.append(None)
        conn.close()
        if None in key_values:
            return None
        return tuple(key_values)
    except sqlite3.Error as e:
        print(f"Error checking database {db_path}: {e}")
        return None

def filter_databases(directory, keys):
    grouped_files = defaultdict(list)
    for filename in os.listdir(directory):
        if filename.endswith(".db") and not filename.startswith("_merged"):
            db_path = os.path.join(directory, filename)
            key_values = check_keys(db_path, keys)
            if key_values:
                grouped_files[key_values].append(db_path)
    return grouped_files

def main():
    parser = argparse.ArgumentParser(description='Filter SQLite .db files based on CONFIG table key values.')
    parser.add_argument('directory', help='The directory containing the .db files.')
    parser.add_argument('--keys', required=True, help='Comma-separated list of keys to filter by.')
    args = parser.parse_args()

    keys = args.keys.split(',')
    grouped_files = filter_databases(args.directory, keys)

    for key_values, files in grouped_files.items():
        print(f"\nCombination {key_values}:")
        for file in files:
            print(file)

if __name__ == '__main__':
    main()
