import dash
from dash import html, dcc
import dash_table
import sqlite3
import os

def load_config_data(db_path):
    config = {}
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM CONFIG")
            rows = cursor.fetchall()

        config = {}
        for key, value in rows:
            if value.isdigit():
                parsed_value = int(value)
            elif all(char.isdigit() or char == '.' for char in value):
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value
            else:
                parsed_value = value

            key_parts = key.split('__')
            d = config
            for part in key_parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[key_parts[-1]] = parsed_value
    except sqlite3.Error:
        config = {}

    return config

app = dash.Dash(__name__)

# Directory containing the database files
db_directory = 'computed/cirrus/'

# List all database files in the directory
db_files = [f for f in os.listdir(db_directory) if f.endswith('.db')]

# Create a layout with two columns
app.layout = html.Div(
    [
        html.Div(
            [
                html.H4(db_file),
                dash_table.DataTable(
                    data=[{"key": key, "value": value} for key, value in load_config_data(os.path.join(db_directory, db_file)).items()],
                    columns=[{"name": "Key", "id": "key"}, {"name": "Value", "id": "value"}],
                    style_table={'overflowX': 'auto'},  # Allow horizontal scrolling
                    style_header={'display': 'none'}   # Hide the header
                ),
                html.A("SEE VISUALIZATION", href=f"http://localhost:8051/{db_file}", target="_blank"),
                html.Br(),
            ],
            style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}
        ) for db_file in db_files
    ],
    style={'display': 'flex', 'flex-wrap': 'wrap'}
)

if __name__ == '__main__':
    app.run_server(debug=True)
