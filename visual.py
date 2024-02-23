from dash import dcc, html, Input, Output
from dash import dcc, html, Input, Output
from plotly.graph_objs import Box
import dash
import dash
import pandas
import pandas as pd  # Ensure pandas is imported correctly
import plotly.graph_objs as go
import plotly.graph_objs as go
import sqlite3
import sqlite3
import yaml
import yaml


# Dash App Setup
app = dash.Dash(__name__)

# Extend the App Layout to include the boxplot
app.layout = html.Div([
    dcc.Graph(id='policy-performance-plot'),
    dcc.Graph(id='episode-length-boxplot'),  # New boxplot graph
    dcc.Graph(id='fitness-lambda-plot'),
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

# Callback for policy performance plot
@app.callback(
    Output('policy-performance-plot', 'figure'),
    [Input('policy-performance-plot', 'id')]
)
def load_policy_performance_data(_):
    db_path = load_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch num_training_episodes and policy_id from policies_info
    cursor.execute('SELECT policy_id, num_training_episodes FROM policies_info')
    training_data = cursor.fetchall()

    # Calculate average episode length for each policy
    avg_episode_lengths = []
    for policy_id, _ in training_data:
        cursor.execute('SELECT AVG(episode_length) FROM episode_info WHERE policy_id = ?', (policy_id,))
        avg_length = cursor.fetchone()[0]
        avg_episode_lengths.append(avg_length if avg_length is not None else 0)

    policy_ids, num_episodes = zip(*training_data)
    conn.close()

    return {
        'data': [go.Scatter(x=num_episodes, y=avg_episode_lengths, mode='lines+markers', name='Average Episode Length',
                            line=dict(color='blue', width=4))],
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
                'data': [go.Scatter(
                    x=[d[0] for d in fitness_lambda_data],
                    y=[d[1] for d in fitness_lambda_data],
                    mode='lines+markers',
                    name='Fitness-Lambda',
                    line=dict(color='blue', width=4)
                )],
                'layout': go.Layout(
                    title=f'Fitness-Lambda Assignment for Policy {policy_id}',
                    xaxis=dict(title='Fitness', gridcolor=stylish_layout['gridcolor'], gridwidth=stylish_layout['gridwidth']),
                    yaxis=dict(title='Lambda', gridcolor=stylish_layout['gridcolor'], gridwidth=stylish_layout['gridwidth']),
                    font=stylish_layout['font'],
                    paper_bgcolor=stylish_layout['paper_bgcolor'],
                    plot_bgcolor=stylish_layout['plot_bgcolor']
                )
            }

        conn.close()

    return {'data': [], 'layout': go.Layout(title='Click on a Policy to View Fitness-Lambda Assignment')}

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
                'data': [Box(y=episode_lengths, boxpoints='all', jitter=0.3, pointpos=-1.8)],
                'layout': go.Layout(
                    title=f'Episode Lengths Boxplot for Policy {policy_id}',
                    yaxis=dict(title='Episode Length'),
                    font=stylish_layout['font'],
                    paper_bgcolor=stylish_layout['paper_bgcolor'],
                    plot_bgcolor=stylish_layout['plot_bgcolor']
                )
            }

        conn.close()

    return {'data': [], 'layout': go.Layout(title='Select a Policy to View Episode Lengths')}

if __name__ == '__main__':
    app.run_server(debug=True)
