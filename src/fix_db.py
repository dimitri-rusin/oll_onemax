import sqlite3
import argparse
import os

def create_sql_dump(db_path):
    sql_dump_path = f"{db_path}.sql"
    os.system(f"sqlite3 {db_path} .dump > {sql_dump_path}")
    return sql_dump_path

def delete_existing_db(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)

def remove_rollback_statements(sql_dump_path):
    with open(sql_dump_path, 'r') as file:
        lines = file.readlines()
    with open(sql_dump_path, 'w') as file:
        for line in lines:
            if 'ROLLBACK;' not in line:
                file.write(line)

def create_database_from_sql(db_path, sql_dump_path):
    try:
        delete_existing_db(db_path)

        # Connect to the new database (this will create the database file)
        conn = sqlite3.connect(db_path)

        # Read the SQL dump file
        with open(sql_dump_path, 'r') as file:
            sql_dump = file.read()

        # Execute the SQL commands in the dump file
        cursor = conn.cursor()
        cursor.executescript(sql_dump)

        # Commit the changes and close the connection
        conn.commit()
        conn.close()

        print("Database created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Delete the SQL dump file
        if os.path.exists(sql_dump_path):
            os.remove(sql_dump_path)
            print(f"Deleted SQL dump file: {sql_dump_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an SQLite database from an SQL dump.')
    parser.add_argument('db_path', type=str, help='The path to the .db file')

    args = parser.parse_args()
    sql_dump_path = create_sql_dump(args.db_path)
    remove_rollback_statements(sql_dump_path)
    create_database_from_sql(args.db_path, sql_dump_path)
