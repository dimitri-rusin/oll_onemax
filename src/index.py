import argparse
from dash.dependencies import MATCH, ALL

import dash
import os
import sqlite3
from dash.dependencies import Input, Output, State, ALL

db_directory = os.getenv('OO__DB_PATH')
screen_directory_path = '/run/screen/'

def list_file_paths():
    file_dict = {}
    try:
        # Iterate through each subdirectory in the given screen_directory_path
        for subdir in os.listdir(screen_directory_path):
            subdir_path = os.path.join(screen_directory_path, subdir)
            if os.path.isdir(subdir_path):
                # Iterate through each file in the subdirectory
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    filename = os.path.basename(file_path)

                    # Find the first period and the last underscore
                    first_period_index = filename.find('.')
                    last_underscore_index = filename.rfind('_')

                    # Extract the key and value based on the found indices
                    if first_period_index != -1 and last_underscore_index != -1:
                        key = filename[first_period_index + 1:last_underscore_index - 1]
                        value = filename[last_underscore_index + 1:]
                        file_dict[key] = int(value)  # Convert port number to integer
    except Exception as e:
        print(f"Error reading files from {screen_directory_path}: {e}")

    return file_dict

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

      if isinstance(parsed_value, (int, float)):
        formatted_value = f"{parsed_value:,}"
      else:
        formatted_value = parsed_value

      key_parts = key.split('__')
      d = config
      for part in key_parts[:-1]:
        if part not in d:
          d[part] = {}
        d = d[part]
      d[key_parts[-1]] = formatted_value
  except sqlite3.Error:
    config = {}

  return config

app = dash.Dash(__name__)

db_files = [f for f in os.listdir(db_directory) if f.endswith('.db')]

# Dictionary for file names to port numbers
file_to_port_dict = list_file_paths()

# Create a layout with two columns
app.layout = dash.html.Div(
  [
    dash.html.Div(
      [
        dash.html.H4(db_file),
        dash.dash_table.DataTable(
          data=[{"key": key, "value": value} for key, value in load_config_data(os.path.join(db_directory, db_file)).items()],
          columns=[{"name": "Key", "id": "key"}, {"name": "Value", "id": "value"}],
          style_table={'overflowX': 'auto'},
          style_header={'display': 'none'},
          active_cell={'row': 0, 'column': 0},  # Initialize with a default active cell
          id={'type': 'dynamic-table', 'index': db_file},
        ),
        dash.html.A(
          "SEE VISUALIZATION",
          href=f"http://localhost:{file_to_port_dict[db_file.replace('.db', '')]}/",
          target="_blank"
        ),
        dash.html.Br(),
      ],
      style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}
    ) for db_file in db_files if db_file.replace('.db', '') in file_to_port_dict
  ] if any(db_file.replace('.db', '') in file_to_port_dict for db_file in db_files) else [
      dash.html.Div([
          dash.html.P(f"{len(db_files)} database files found. Please launch screen sessions, first."),
          dash.html.Ul([dash.html.Li(db_file) for db_file in db_files])
      ])
  ],
  style={'display': 'flex', 'flex-wrap': 'wrap'}
)



@app.callback(
    Output({'type': 'dynamic-table', 'index': MATCH}, 'style_data_conditional'),
    [Input({'type': 'dynamic-table', 'index': MATCH}, 'active_cell')]
)
def highlight_row(active_cell):
    if not active_cell:
        return []

    highlight_color = '#D2F3FF'  # Your chosen highlight color

    return [
        {
            'if': {'row_index': active_cell['row']},
            'backgroundColor': highlight_color,
            'color': 'black'
        },
        {
            'if': {'state': 'active'},  # This targets the active cell
            'backgroundColor': highlight_color,
            'border': 'none'  # Removing the border by setting it to 'none'
        },
        {
            'if': {'state': 'selected'},  # This targets selected cells
            'backgroundColor': highlight_color,
            'border': 'none'  # Removing the border by setting it to 'none'
        },
    ]


if __name__ == '__main__':
  # Parse command line arguments for the port
  parser = argparse.ArgumentParser(description='Run Dash app')
  parser.add_argument('--port', type=int, default=8050, help='Port to run the Dash app on')
  args = parser.parse_args()

  # Run the Dash app on the specified port
  app.run_server(debug=True, port=args.port)
