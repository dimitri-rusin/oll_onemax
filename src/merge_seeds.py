import argparse
import filter_dbs
import fix_db
import os
import sqlite3



def get_db_path(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT value FROM CONFIG WHERE key='database_path'")
    result = cursor.fetchone()
    return result[0] if result else None

def get_table_schema(connection, table_name):
    cursor = connection.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    schema = cursor.fetchall()
    return schema

def create_new_database(new_db_path, schemas):
    # Remove the existing database file if it exists
    if os.path.exists(new_db_path):
        os.remove(new_db_path)

    conn = sqlite3.connect(new_db_path)
    cursor = conn.cursor()

    for table_name, schema in schemas.items():
        if table_name == 'EVALUATION_EPISODES':
            columns = ', '.join([f"{col[1]} {col[2].replace('PRIMARY KEY', '').replace('UNIQUE', '')}" for col in schema])
        else:
            columns = 'database_path TEXT, ' + ', '.join([f"{col[1]} {col[2].replace('PRIMARY KEY', '').replace('UNIQUE', '')}" for col in schema])
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        cursor.execute(create_table_query)

    conn.commit()
    return conn

def copy_and_extend_table(source_conn, dest_conn, table_name, db_path_value):
    source_cursor = source_conn.cursor()
    dest_cursor = dest_conn.cursor()

    source_cursor.execute(f"SELECT * FROM {table_name}")
    rows = source_cursor.fetchall()

    columns = [desc[0] for desc in source_cursor.description]
    if table_name != 'EVALUATION_EPISODES':
        columns.insert(0, "database_path")

    columns_str = ', '.join(columns)
    placeholders = ', '.join(['?' for _ in columns])

    for row in rows:
        if table_name == 'EVALUATION_EPISODES':
            row_with_db_path = list(row)
        else:
            row_with_db_path = [db_path_value] + list(row)

        try:
            dest_cursor.execute(f'''
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
            ''', row_with_db_path)
        except sqlite3.IntegrityError as e:
            print(f"IntegrityError: {e} - Skipping row {row_with_db_path}")

    dest_conn.commit()

def merged_database(db_paths, new_db_path):
    directory_path = os.path.dirname(new_db_path)
    os.makedirs(directory_path, exist_ok=True)

    assert len(db_paths) > 0
    first_db_path = db_paths[0]
    directory_path = os.path.dirname(first_db_path)

    connections = [sqlite3.connect(database_path) for database_path in db_paths]

    new_conn = None
    failed_path = None
    try:
        # Output the assignment statements
        for i, db_path_value in enumerate(db_paths):
            if db_path_value:
                # print(f"db_path_{i+1} = '{db_path_value}'")
                pass
            else:
                print(f"db_path_{i+1} not found in CONFIG table.")

        # Get table schemas from the first database
        tables = ['CONFIG', 'CONSTRUCTED_POLICIES', 'EVALUATION_EPISODES', 'POLICY_DETAILS']
        schemas = {table: get_table_schema(connections[0], table) for table in tables}

        # Create a new database with tables using the fetched schemas and an additional database_path column where needed
        new_conn = create_new_database(new_db_path, schemas)

        # Copy and extend the tables from each database
        for conn, db_path_value in zip(connections, db_paths):
            if db_path_value:
                for table in tables:
                    copy_and_extend_table(conn, new_conn, table, db_path_value)

    except sqlite3.Error as e:
        print(f"SQLite error (with {db_path_value}): {e}")
        failed_path = db_path_value
    finally:
        # Close the connections
        for conn in connections:
            conn.close()
        if new_conn:
            new_conn.close()

    return failed_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge SQLite databases from a specified folder.')
    parser.add_argument('folder_path', type=str, help='The path to the folder containing .db files')
    parser.add_argument('--keys', required=True, help='Comma-separated list of keys to filter by.')

    args = parser.parse_args()
    folder_path = args.folder_path

    keys = args.keys.split(',')
    grouped_files = filter_dbs.filter_databases(folder_path, keys)

    for key_values, files in grouped_files.items():
      print(key_values)
      print(files)

      failed_path = merged_database(files, os.path.join(folder_path, f'_merged_{str(key_values)}.db'))
      while failed_path is not None:

        sql_dump_path = fix_db.create_sql_dump(failed_path)
        fix_db.remove_rollback_statements(sql_dump_path)
        fix_db.create_database_from_sql(failed_path, sql_dump_path)

        failed_path = merged_database(files, os.path.join(folder_path, f'_merged_{str(key_values)}.db'))
