import sqlite3
import argparse

def rename_column(db_path, old_column_name, new_column_name):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            
            # Fetch column names for the table
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            column_names = [column[1] for column in columns]

            if old_column_name in column_names:
                # Create a list of new column names
                new_columns = [new_column_name if col == old_column_name else col for col in column_names]

                # Create a temporary table with new column names
                cursor.execute(f"CREATE TABLE temp_table ({', '.join(new_columns)});")
                
                # Copy data from old table to new table
                cursor.execute(f"INSERT INTO temp_table SELECT * FROM {table_name};")
                
                # Drop the old table
                cursor.execute(f"DROP TABLE {table_name};")
                
                # Rename temporary table to the original table name
                cursor.execute(f"ALTER TABLE temp_table RENAME TO {table_name};")
                
                print(f"Renamed column {old_column_name} to {new_column_name} in table {table_name}")

        conn.commit()
    except sqlite3.Error as error:
        print(f"Error occurred: {error}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename a column in an SQLite database.')
    parser.add_argument('db_path', type=str, help='Path to the SQLite database.')
    parser.add_argument('--old_column', type=str, default='db_path', help='Name of the column to rename.')
    parser.add_argument('--new_column', type=str, default='database_path', help='New name of the column.')

    args = parser.parse_args()

    rename_column(args.db_path, args.old_column, args.new_column)
