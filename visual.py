from dash import dcc, html, Input, Output
import math
import dash
import inspectify
import pandas
import plotly.graph_objs
import plotly.graph_objs as go
import sqlite3
import yaml

config = None

# Dash App Setup
app = dash.Dash(__name__)

app.title = 'Tuning OLL'

app.layout = html.Div([
  html.Div([
    dcc.Graph(id='policy-performance-plot'),
    dcc.Graph(id='fitness-lambda-plot'),
    dcc.Checklist(
      id='auto-update-switch',
      options=[
        {'label': 'Auto Update Plot Every 5 Seconds', 'value': 'ON'}
      ],
      value=['ON'],  # Default value set to 'ON'
      style={'fontFamily': 'Courier New, monospace', 'color': 'RebeccaPurple'}
    ),
    dcc.Interval(
      id='interval-component',
      interval=5*1000,  # in milliseconds
      n_intervals=0
    )
  ], style={'display': 'inline-block', 'width': '70%'}),  # Adjust width as needed

  html.Div([
    dcc.Dropdown(
      id='xaxis-selector',
      options=[
        {'label': 'Number of Training Episodes', 'value': 'num_training_episodes'},
        {'label': 'Number of Q-Table Updates', 'value': 'num_q_table_updates'},
        {'label': 'Number of Total Timesteps', 'value': 'num_total_timesteps'},  # New option added
      ],
      value='num_total_timesteps',  # Default value
      style={'width': '100%'}
    ),
  ], style={'display': 'inline-block', 'width': '25%', 'vertical-align': 'top'}),  # Adjust width and alignment as needed

], style={'fontFamily': 'Courier New, monospace', 'backgroundColor': 'rgba(0,0,0,0)'})

# Load database path from .env.yaml
def load_db_path():
  with open(".env.yaml") as file:
    global config
    config = yaml.safe_load(file)
  return config['db_path']

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
  cursor.execute(f'SELECT policy_id, {x_axis_sql_column} FROM policies_info WHERE policy_id >= 1')
  training_data = cursor.fetchall()

  # Update the mapping of policy IDs to their x-values to include num_total_timesteps
  policy_id_to_x_values = {policy_id: {column_name: value for column_name, value in zip(['num_training_episodes', 'num_q_table_updates', 'num_total_timesteps'], row)}
               for policy_id, *row in cursor.execute('SELECT policy_id, num_training_episodes, num_q_table_updates, num_total_timesteps FROM policies_info WHERE policy_id >= 1')}






  # Calculate average number of function evaluations and standard deviation for each policy
  avg_function_evaluations = []
  std_dev_evaluations = []
  for policy_id, _ in training_data:
    cursor.execute('SELECT num_function_evaluations FROM episode_info WHERE policy_id = ?', (policy_id,))
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
  cursor.execute('SELECT AVG(num_function_evaluations) FROM episode_info WHERE policy_id = -1')
  baseline_result = cursor.fetchone()
  baseline_avg_length = baseline_result[0] if baseline_result else 0

  # Determine if a point has been clicked and find the corresponding policy ID
  selected_point = None
  if clickData:
    point_index = clickData['points'][0]['pointIndex']
    if point_index in policy_id_to_x_values:
      policy_id = list(policy_id_to_x_values.keys())[point_index]  # Retrieve policy ID based on point index
      new_x_value = policy_id_to_x_values[policy_id][xaxis_choice]  # Get new x value based on xaxis choice

      # Create the selected point with the new x value
      selected_point = go.Scatter(x=[new_x_value], y=[clickData['points'][0]['y']], mode='markers', marker=dict(color='red', size=15), name='Selected Point')

  conn.close()

  # Create plots for average FEs and standard deviation
  data = [
    go.Scatter(x=num_episodes, y=avg_function_evaluations, mode='lines+markers', name='#FEs until optimum', line=dict(color='blue', width=4)),
    go.Scatter(x=num_episodes, y=[avg + std for avg, std in zip(avg_function_evaluations, std_dev_evaluations)], mode='lines', line=dict(color='rgba(173,216,230,0.2)'), name='Upper Bound (Mean + Std. Dev.)'),
    go.Scatter(x=num_episodes, y=[avg - std for avg, std in zip(avg_function_evaluations, std_dev_evaluations)], mode='lines', fill='tonexty', line=dict(color='rgba(173,216,230,0.2)'), name='Lower Bound (Mean - Std. Dev.)'),
    go.Scatter(x=[min(num_episodes), max(num_episodes)] if num_episodes else [0], y=[baseline_avg_length, baseline_avg_length], mode='lines', name='Theory: √(𝑛/(𝑛 − 𝑓(𝑥)))', line=dict(color='orange', width=2, dash='dash'))
  ]

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
        tickformat=',',  # Add thousands separator
      ),
      yaxis=dict(
        title='#FEs until optimum',  # Updated Y-axis title
        gridcolor=stylish_layout['gridcolor'],
        gridwidth=stylish_layout['gridwidth'],
        tickformat=',',  # Add thousands separator
      ),
      font=stylish_layout['font'],
      paper_bgcolor=stylish_layout['paper_bgcolor'],
      plot_bgcolor=stylish_layout['plot_bgcolor']
    )
  }

# Updated callback for fitness-lambda plot
@app.callback(
  Output('fitness-lambda-plot', 'figure'),
  [Input('policy-performance-plot', 'clickData'),
   Input('xaxis-selector', 'value')]  # Add the x-axis selector input
)
def update_fitness_lambda_plot(clickData, xaxis_choice):
  global policy_id_to_x_values

  if clickData:
    point_index = clickData['points'][0]['pointIndex']
    if point_index in policy_id_to_x_values:
      policy_id = list(policy_id_to_x_values.keys())[point_index]  # Retrieve policy ID based on point index

      db_path = load_db_path()
      conn = sqlite3.connect(db_path)
      cursor = conn.cursor()

      # Fetch fitness-lambda data for the selected policy
      cursor.execute('SELECT fitness, lambda FROM policies_data WHERE policy_id = ?', (policy_id,))
      fitness_lambda_data = cursor.fetchall()

      # Fitness-Lambda plot with connecting lines
      conn.close()
      return {
        'data': [plotly.graph_objs.Scatter(
          x=[d[0] for d in fitness_lambda_data],
          y=[d[1] + 1 for d in fitness_lambda_data], # + 1, because we save the lambda - 1 value, but present here the lambda
          mode='lines+markers',
          name='Fitness-Lambda',
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

  return {'data': [], 'layout': plotly.graph_objs.Layout(title='Click on a Policy to View Fitness-Lambda Assignment')}



if __name__ == '__main__':
  app.run_server(debug=True)
