import sqlite3
import os

def get_db_path(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT value FROM CONFIG WHERE key='db_path'")
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
            columns = 'db_path TEXT, ' + ', '.join([f"{col[1]} {col[2].replace('PRIMARY KEY', '').replace('UNIQUE', '')}" for col in schema])
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
        columns.insert(0, "db_path")

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

def main():
    # Paths to the two SQLite databases
    db_path_1 = '/home/dimitri/code/oll_onemax/computed/cirrus-login2/four_one_specific/abdominal.db'
    db_path_2 = '/home/dimitri/code/oll_onemax/computed/cirrus-login2/four_one_specific/abide.db'
    new_db_path = 'src/merged.db'

    # Create connections to the two databases
    conn1 = sqlite3.connect(db_path_1)
    conn2 = sqlite3.connect(db_path_2)

    new_conn = None
    try:
        # Get the db_path value from the CONFIG table for each connection
        db_path_value_1 = get_db_path(conn1)
        db_path_value_2 = get_db_path(conn2)

        # Output the assignment statements
        if db_path_value_1:
            print(f"db_path_1 = '{db_path_value_1}'")
        else:
            print("db_path_1 not found in CONFIG table.")

        if db_path_value_2:
            print(f"db_path_2 = '{db_path_value_2}'")
        else:
            print("db_path_2 not found in CONFIG table.")

        # Get table schemas from the first database
        tables = ['CONFIG', 'CONSTRUCTED_POLICIES', 'EVALUATION_EPISODES', 'POLICY_DETAILS']
        schemas = {table: get_table_schema(conn1, table) for table in tables}

        # Create a new database with tables using the fetched schemas and an additional db_path column where needed
        new_conn = create_new_database(new_db_path, schemas)

        # Copy and extend the tables from the first database
        if db_path_value_1:
            for table in tables:
                copy_and_extend_table(conn1, new_conn, table, db_path_value_1)

        # Copy and extend the tables from the second database
        if db_path_value_2:
            for table in tables:
                copy_and_extend_table(conn2, new_conn, table, db_path_value_2)

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        # Close the connections
        conn1.close()
        conn2.close()
        if new_conn:
            new_conn.close()

if __name__ == '__main__':
    main()
