from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash
import inspectify
import math
import os
import pandas
import plotly
import plotly.graph_objs as go
import sqlite3
import yaml

env_yaml_path = '.env.yaml'
config = None

app = dash.Dash(__name__)
app.title = 'Tuning OLL'



def load_db_path():
  return os.getenv('OO_DB_PATH')

def load_config_data():
  db_path = load_db_path()  # Ensure this function is defined elsewhere to get the database path
  config = []

  with sqlite3.connect(db_path) as conn:
      cursor = conn.cursor()
      cursor.execute("SELECT key, value FROM CONFIG")
      rows = cursor.fetchall()

  # Process each row to infer the type and construct a nested dictionary
  config = {}
  for key, value in rows:
      # Infer the type
      if value.isdigit():
          parsed_value = int(value)
      elif all(char.isdigit() or char == '.' for char in value):
          try:
              parsed_value = float(value)
          except ValueError:
              parsed_value = value
      else:
          parsed_value = value

      # Create nested dictionaries based on key structure
      key_parts = key.split('__')
      d = config
      for part in key_parts[:-1]:
          if part not in d:
              d[part] = {}
          d = d[part]
      d[key_parts[-1]] = parsed_value

  return config

config = load_config_data()


def flatten_config(config, parent_key=''):
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}__{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key))
        else:
            items.append({'Key': new_key, 'Value': v})
    return items

flattened_config = flatten_config(config)


app.layout = html.Div([
    html.Div([
        dcc.Graph(id='policy-performance-plot'),
        dcc.Graph(id='fitness-lambda-plot'),
        dcc.Checklist(
            id='auto-update-switch',
            options=[
                {'label': 'Auto Update Plot Every 5 Seconds', 'value': 'ON'}
            ],
            value=['ON'],
            style={'fontFamily': 'Courier New, monospace', 'color': 'RebeccaPurple'}
        ),
        dcc.Interval(
            id='interval-component',
            interval=5*1000,
            n_intervals=0
        )
    ], style={'display': 'inline-block', 'width': '70%'}),

    html.Div([
        dcc.Dropdown(
            id='xaxis-selector',
            options=[
                {'label': 'Number of Training Episodes', 'value': 'num_training_episodes'},
                {'label': 'Number of Total Function Evaluations', 'value': 'num_total_function_evaluations'},
                {'label': 'Number of Total Timesteps', 'value': 'num_total_timesteps'},
            ],
            value='num_total_timesteps',
            style={'width': '100%'}
        ),
    ], style={'display': 'inline-block', 'width': '25%', 'vertical-align': 'top'}),

    # DataTable for displaying config key-value pairs below the Fitness-Lambda plot
    dash_table.DataTable(
        id='config-table',
        columns=[{"name": i, "id": i} for i in ['Key', 'Value']],
        data=flattened_config,
        style_cell={'textAlign': 'left'},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        }
    ),



], style={'fontFamily': 'Courier New, monospace', 'backgroundColor': 'rgba(0,0,0,0)'})




def did_we_click_the_mean_policy_curve(clickData):
  # This value is 0, only because we insert the policy curve at the first position of the data list below.
  return clickData['points'][0]['curveNumber'] == 0

@app.callback(
    Output('config-table', 'data'),
    [Input('policy-performance-plot', 'clickData')]
)
def update_config_table(clickData):
    if clickData is None or not did_we_click_the_mean_policy_curve(clickData):
        raise dash.exceptions.PreventUpdate

    # Extracting x and y values from the clicked point
    x_value = clickData['points'][0]['x']
    y_value = clickData['points'][0]['y']


    point_index = clickData['points'][0]['pointIndex']
    policy_id = list(policy_id_to_x_values.keys())[point_index]


    # You can format these values and add them to your config data
    updated_config_data = flattened_config.copy()
    updated_config_data.append({'Key': 'Selected Policy X Value', 'Value': f"{x_value:,}"})
    updated_config_data.append({'Key': 'Selected Policy Y Value', 'Value': f"{y_value:,}"})
    updated_config_data.append({'Key': 'Policy ID', 'Value': f"{policy_id:,}"})

    return updated_config_data









# Stylish layout for plots
stylish_layout = {
  'title': {'x': 0.5},
  'font': {'family': 'Courier New, monospace', 'size': 18, 'color': 'RebeccaPurple'},
  'paper_bgcolor': 'rgba(0,0,0,0)',
  'plot_bgcolor': 'rgba(245, 245, 245, 1)',
  'gridcolor': 'LightPink',
  'gridwidth': 1
}

# Global variable to store the mapping of policy IDs to their x-values
policy_id_to_x_values = {}







# Updated Callback for policy performance plot
@app.callback(
    Output('policy-performance-plot', 'figure'),
    [Input('policy-performance-plot', 'clickData'),
     Input('interval-component', 'n_intervals'),
     Input('auto-update-switch', 'value'),
     Input('xaxis-selector', 'value')]
)
def load_policy_performance_data(clickData, n_intervals, auto_update_value, xaxis_choice):
  global policy_id_to_x_values
  # Only update if auto-update is switched on
  if 'ON' not in auto_update_value:
    raise dash.exceptions.PreventUpdate




  db_path = load_db_path()
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()



  # Modify the SQL query based on the x-axis choice
  x_axis_sql_column = xaxis_choice
  cursor.execute(f'SELECT policy_id, {x_axis_sql_column} FROM CONSTRUCTED_POLICIES WHERE policy_id >= 1')
  training_data = cursor.fetchall()

  # Update the mapping of policy IDs to their x-values to include num_total_timesteps
  policy_id_to_x_values = {policy_id: {column_name: value for column_name, value in zip(['num_training_episodes', 'num_total_function_evaluations', 'num_total_timesteps'], row)}
               for policy_id, *row in cursor.execute('SELECT policy_id, num_training_episodes, num_total_function_evaluations, num_total_timesteps FROM CONSTRUCTED_POLICIES WHERE policy_id >= 1')}


  # Calculate average number of function evaluations and standard deviation for each policy
  avg_function_evaluations = []
  std_dev_evaluations = []
  for policy_id, _ in training_data:
    cursor.execute('SELECT num_function_evaluations FROM EVALUATION_EPISODES WHERE policy_id = ?', (policy_id,))
    evaluations = [e[0] for e in cursor.fetchall()]

    # Calculate average
    if len(evaluations) >= config['num_evaluation_episodes']:
      avg_evaluations = sum(evaluations) / len(evaluations)
      # Calculate standard deviation
      std_dev = math.sqrt(sum((e - avg_evaluations) ** 2 for e in evaluations) / len(evaluations))
    else:
      avg_evaluations = None
      std_dev = 0

    avg_function_evaluations.append(avg_evaluations if avg_evaluations is not None else 0)
    std_dev_evaluations.append(std_dev)




  policy_ids, num_episodes = zip(*training_data) if training_data else ([], [])

  # Fetch baseline average episode length
  cursor.execute('SELECT AVG(num_function_evaluations) FROM EVALUATION_EPISODES WHERE policy_id = -1')
  baseline_result = cursor.fetchone()
  baseline_avg_length = baseline_result[0] if baseline_result else 0





  # Fetch baseline average episode length
  cursor.execute('SELECT num_function_evaluations FROM EVALUATION_EPISODES WHERE policy_id = -1')
  baseline_evaluations = [e[0] for e in cursor.fetchall()]
  baseline_avg_length = sum(baseline_evaluations) / len(baseline_evaluations) if baseline_evaluations else 0

  # Calculate the variance for baseline evaluations
  baseline_variance = sum((e - baseline_avg_length) ** 2 for e in baseline_evaluations) / (len(baseline_evaluations) - 1) if len(baseline_evaluations) > 1 else 0
  baseline_std_dev = math.sqrt(baseline_variance)

  # Calculate upper and lower bounds for the baseline
  baseline_upper_bound = [baseline_avg_length + baseline_std_dev] * len(num_episodes)
  baseline_lower_bound = [baseline_avg_length - baseline_std_dev] * len(num_episodes)

  data = []

  # Determine if a point has been clicked and find the corresponding policy ID
  selected_point = None
  if clickData and did_we_click_the_mean_policy_curve(clickData):
    point_index = clickData['points'][0]['pointIndex']
    # Check if the point index exists in policy_id_to_x_values
    policy_id = list(policy_id_to_x_values.keys())[point_index]  # Retrieve policy ID based on point index
    if policy_id in policy_id_to_x_values:
      new_x_value = policy_id_to_x_values[policy_id][xaxis_choice]  # Get new x value based on xaxis choice

      # Create the selected point with the new x value
      selected_point = go.Scatter(
        x=[new_x_value],
        y=[clickData['points'][0]['y']],
        mode='markers',
        marker=dict(color='red', size=15),
        name='Selected Point'
      )

  conn.close()

  # Create plots for average FEs and standard deviation
  data = [
    go.Scatter(x=num_episodes, y=avg_function_evaluations, mode='lines+markers', name='#FEs until optimum', line=dict(color='blue', width=4)),
    go.Scatter(x=num_episodes, y=[avg + std for avg, std in zip(avg_function_evaluations, std_dev_evaluations)], mode='lines', line=dict(color='rgba(173,216,230,0.2)'), name='Upper Bound (Mean + Std. Dev.)'),
    go.Scatter(x=num_episodes, y=[avg - std for avg, std in zip(avg_function_evaluations, std_dev_evaluations)], mode='lines', fill='tonexty', line=dict(color='rgba(173,216,230,0.2)'), name='Lower Bound (Mean - Std. Dev.)'),
    go.Scatter(x=[min(num_episodes), max(num_episodes)] if num_episodes else [0], y=[baseline_avg_length, baseline_avg_length], mode='lines', name='Theory: √(𝑛/(𝑛 − 𝑓(𝑥)))', line=dict(color='orange', width=2, dash='dash'))
  ]


  # Add the shadowed variance for the baseline to the plot
  data.extend([
      go.Scatter(
        x=num_episodes,
        y=baseline_upper_bound,
        mode='lines',
        line=dict(color='rgba(255, 165, 0, 0.2)'),
        name='Upper Bound (Baseline Variance)'
      ),
      go.Scatter(
        x=num_episodes,
        y=baseline_lower_bound,
        mode='lines',
        fill='tonexty',
        line=dict(color='rgba(255, 165, 0, 0.2)'),
        name='Lower Bound (Baseline Variance)'
      )
  ])

  if selected_point:
    data.append(selected_point)

  return {
    'data': data,
    'layout': go.Layout(
      title='Policy Performance Plot',
      xaxis=dict(
        title=xaxis_choice.replace('_', ' ').title(),
        gridcolor=stylish_layout['gridcolor'],
        gridwidth=stylish_layout['gridwidth'],
        tickformat=',',
      ),
      yaxis=dict(
        title='#FEs until optimum',
        gridcolor=stylish_layout['gridcolor'],
        gridwidth=stylish_layout['gridwidth'],
        tickformat=',',
      ),
      font=stylish_layout['font'],
      paper_bgcolor=stylish_layout['paper_bgcolor'],
      plot_bgcolor=stylish_layout['plot_bgcolor']
    )
  }



@app.callback(
  Output('fitness-lambda-plot', 'figure'),
  [Input('policy-performance-plot', 'clickData'),
   Input('xaxis-selector', 'value')]
)
def update_fitness_lambda_plot(clickData, xaxis_choice):
  global policy_id_to_x_values

  db_path = load_db_path()
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Fetch baseline fitness-lambda data (policy_id = -1)
  cursor.execute('SELECT fitness, lambda_minus_one FROM POLICY_DETAILS WHERE policy_id = -1')
  baseline_fitness_lambda_data = cursor.fetchall()

  baseline_curve = plotly.graph_objs.Scatter(
      x=[d[0] for d in baseline_fitness_lambda_data],
      y=[d[1] + 1 for d in baseline_fitness_lambda_data],
      mode='lines+markers',
      name='Baseline Fitness-Lambda',
      line=dict(color='orange', width=4)
  )

  # Data for the selected policy
  selected_policy_curve = {'data': [], 'layout': {}}
  if clickData and did_we_click_the_mean_policy_curve(clickData):
    point_index = clickData['points'][0]['pointIndex']
    policy_id = list(policy_id_to_x_values.keys())[point_index]
    if policy_id in policy_id_to_x_values:

      # Fetch mean and variance of initial fitness for the selected policy
      cursor.execute('SELECT mean_initial_fitness, variance_initial_fitness FROM CONSTRUCTED_POLICIES WHERE policy_id = ?', (policy_id,))
      fitness_stats = cursor.fetchone()
      mean_initial_fitness = fitness_stats[0]
      variance_initial_fitness = fitness_stats[1]
      std_dev_initial_fitness = math.sqrt(variance_initial_fitness)

      # Fetch fitness-lambda data for the selected policy
      cursor.execute('SELECT fitness, lambda_minus_one FROM POLICY_DETAILS WHERE policy_id = ?', (policy_id,))
      fitness_lambda_data = cursor.fetchall()

      selected_policy_curve = {
        'data': [plotly.graph_objs.Scatter(
          x=[d[0] for d in fitness_lambda_data],
          y=[d[1] + 1 for d in fitness_lambda_data],
          mode='lines+markers',
          name=f'Fitness-Lambda Policy {policy_id}',
          line=dict(color='blue', width=4)
        )],
        'layout': plotly.graph_objs.Layout(
          title=f'Fitness-Lambda Assignment for Policy {policy_id}',
          xaxis=dict(title='Fitness', gridcolor=stylish_layout['gridcolor'], gridwidth=stylish_layout['gridwidth']),
          yaxis=dict(title='Lambda', gridcolor=stylish_layout['gridcolor'], gridwidth=stylish_layout['gridwidth']),
          font=stylish_layout['font'],
          paper_bgcolor=stylish_layout['paper_bgcolor'],
          plot_bgcolor=stylish_layout['plot_bgcolor']
        )
      }


      # Add shaded area for variance
      # Upper bound line (mean + std_dev)
      selected_policy_curve['data'].append(
        go.Scatter(
          x=[mean_initial_fitness + std_dev_initial_fitness] * 2,
          y=[0, max([d[1] + 1 for d in fitness_lambda_data])],  # Adjust Y-axis limits
          mode='lines',
          line=dict(width=0),
          showlegend=False
        )
      )

      # Lower bound line (mean - std_dev)
      selected_policy_curve['data'].append(
        go.Scatter(
          x=[mean_initial_fitness - std_dev_initial_fitness] * 2,
          y=[0, max([d[1] + 1 for d in fitness_lambda_data])],  # Adjust Y-axis limits
          mode='lines',
          fill='tonexty',
          fillcolor='rgba(0, 255, 0, 0.2)',  # Semi-transparent green for the variance
          line=dict(width=0),
          name='Variance Initial Fitness'
        )
      )

      # Vertical line for the mean
      selected_policy_curve['data'].append(
        go.Scatter(
          x=[mean_initial_fitness, mean_initial_fitness],
          y=[0, max([d[1] + 1 for d in fitness_lambda_data])],  # Adjust Y-axis limits
          mode='lines',
          name=f'Mean Initial Fitness',
          line=dict(color='green', width=2, dash='dot')
        )
      )

  conn.close()

  # Combine data from selected policy and baseline
  all_data = [baseline_curve] + selected_policy_curve['data']

  # Use layout from selected policy curve if it exists, else use default
  layout = selected_policy_curve['layout'] if selected_policy_curve['data'] else {
      'title': 'Fitness-Lambda Assignment',
      'xaxis': {'title': 'Fitness'},
      'yaxis': {'title': 'Lambda'},
      'font': stylish_layout['font'],
      'paper_bgcolor': stylish_layout['paper_bgcolor'],
      'plot_bgcolor': stylish_layout['plot_bgcolor']
  }

  return {'data': all_data, 'layout': layout}















if __name__ == '__main__':
  app.run_server(debug=True)
