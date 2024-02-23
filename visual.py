from dash import dcc, html, Input, Output
import dash
import pandas
import plotly.graph_objs
import plotly.graph_objs as go
import sqlite3
import yaml

# Dash App Setup
app = dash.Dash(__name__)

app.title = 'Tuning OLL'

# Extend the App Layout to include the boxplot and interval component
app.layout = html.Div([
    dcc.Graph(id='policy-performance-plot'),
    dcc.Graph(id='episode-length-boxplot'),  # New boxplot graph
    dcc.Graph(id='fitness-lambda-plot'),
    dcc.Checklist(
        id='auto-update-switch',
        options=[
            {'label': 'Auto Update Plot Every 5 Seconds', 'value': 'ON'}
        ],
        value=[],
        style={'fontFamily': 'Courier New, monospace', 'color': 'RebeccaPurple'}
    ),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds
        n_intervals=0
    )
], style={'fontFamily': 'Courier New, monospace', 'backgroundColor': 'rgba(0,0,0,0)'})




# Load database path from .env.yaml
def load_db_path():
    with open(".env.yaml") as file:
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


# Callback for policy performance plot with clickData and interval component
@app.callback(
    Output('policy-performance-plot', 'figure'),
    [Input('policy-performance-plot', 'clickData'),
     Input('interval-component', 'n_intervals'),
     Input('auto-update-switch', 'value')]
)
def load_policy_performance_data(clickData, n_intervals, auto_update_value):
    # Only update if auto-update is switched on
    if 'ON' not in auto_update_value:
        raise dash.exceptions.PreventUpdate

    db_path = load_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch num_training_episodes and policy_id from policies_info for policy_id >= 1
    cursor.execute('SELECT policy_id, num_training_episodes FROM policies_info WHERE policy_id >= 1')
    training_data = cursor.fetchall()

    # Calculate average episode length for each policy
    avg_episode_lengths = []
    for policy_id, _ in training_data:
        cursor.execute('SELECT AVG(episode_length) FROM episode_info WHERE policy_id = ?', (policy_id,))
        avg_length = cursor.fetchone()[0]
        avg_episode_lengths.append(avg_length if avg_length is not None else 0)

    policy_ids, num_episodes = zip(*training_data) if training_data else ([], [])

    # Fetch baseline average episode length
    cursor.execute('SELECT AVG(episode_length) FROM episode_info WHERE policy_id = -1')
    baseline_result = cursor.fetchone()
    baseline_avg_length = baseline_result[0] if baseline_result else 0

    # Check if a point has been clicked
    selected_point = None
    if clickData:
        selected_x = clickData['points'][0]['x']
        selected_y = clickData['points'][0]['y']
        selected_point = go.Scatter(x=[selected_x], y=[selected_y], mode='markers', marker=dict(color='red', size=15), name='Selected Point')

    conn.close()

    data = [
        go.Scatter(x=num_episodes, y=avg_episode_lengths, mode='lines+markers', name='Average Episode Length', line=dict(color='blue', width=4)),
        go.Scatter(x=[min(num_episodes), max(num_episodes)] if num_episodes else [0], y=[baseline_avg_length, baseline_avg_length], mode='lines', name='Baseline Policy', line=dict(color='orange', width=2, dash='dash'))
    ]

    if selected_point:
        data.append(selected_point)

    return {
        'data': data,
        'layout': go.Layout(
            title='Policy Performance Plot',
            xaxis=dict(title='Number of Training Episodes', gridcolor=stylish_layout['gridcolor'], gridwidth=stylish_layout['gridwidth']),
            yaxis=dict(title='Average Episode Length', gridcolor=stylish_layout['gridcolor'], gridwidth=stylish_layout['gridwidth']),
            font=stylish_layout['font'],
            paper_bgcolor=stylish_layout['paper_bgcolor'],
            plot_bgcolor=stylish_layout['plot_bgcolor']
        )
    }



# Callback for fitness-lambda plot
@app.callback(
    Output('fitness-lambda-plot', 'figure'),
    [Input('policy-performance-plot', 'clickData')]
)
def update_fitness_lambda_plot(clickData):
    if clickData:
        num_training_episodes = clickData['points'][0]['x']  # Extract the number of training episodes
        db_path = load_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Find the policy ID corresponding to the clicked number of training episodes
        cursor.execute('SELECT policy_id FROM policies_info WHERE num_training_episodes = ?', (num_training_episodes,))
        result = cursor.fetchone()
        policy_id = result[0] if result else None

        if policy_id is not None:
            # Fetch fitness-lambda data
            cursor.execute('SELECT fitness, lambda FROM policies_data WHERE policy_id = ?', (policy_id,))
            fitness_lambda_data = cursor.fetchall()

            # Fitness-Lambda plot with connecting lines
            conn.close()
            return {
                'data': [plotly.graph_objs.Scatter(
                    x=[d[0] for d in fitness_lambda_data],
                    y=[d[1] for d in fitness_lambda_data],
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

        conn.close()

    return {'data': [], 'layout': plotly.graph_objs.Layout(title='Click on a Policy to View Fitness-Lambda Assignment')}

# Callback for the episode length boxplot
@app.callback(
    Output('episode-length-boxplot', 'figure'),
    [Input('policy-performance-plot', 'clickData')]
)
def update_episode_length_boxplot(clickData):
    if clickData:
        num_training_episodes = clickData['points'][0]['x']
        db_path = load_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Find the policy ID corresponding to the clicked number of training episodes
        cursor.execute('SELECT policy_id FROM policies_info WHERE num_training_episodes = ?', (num_training_episodes,))
        result = cursor.fetchone()
        policy_id = result[0] if result else None

        if policy_id is not None:
            # Fetch episode lengths for the selected policy
            cursor.execute('SELECT episode_length FROM episode_info WHERE policy_id = ?', (policy_id,))
            episode_lengths = cursor.fetchall()
            episode_lengths = [length[0] for length in episode_lengths]  # Extract lengths from tuples

            conn.close()
            return {
                'data': [plotly.graph_objs.Box(y=episode_lengths, boxpoints='all', jitter=0.3, pointpos=-1.8)],
                'layout': plotly.graph_objs.Layout(
                    title=f'Episode Lengths Boxplot for Policy {policy_id}',
                    yaxis=dict(title='Episode Length'),
                    font=stylish_layout['font'],
                    paper_bgcolor=stylish_layout['paper_bgcolor'],
                    plot_bgcolor=stylish_layout['plot_bgcolor']
                )
            }

        conn.close()

    return {'data': [], 'layout': plotly.graph_objs.Layout(title='Select a Policy to View Episode Lengths')}

if __name__ == '__main__':
    app.run_server(debug=True)
