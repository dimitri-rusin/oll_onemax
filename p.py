import sqlite3
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def fetch_data(db_path):
    """Fetch episode lengths and training episodes from the database."""
    with sqlite3.connect(db_path) as conn:
        query = '''
        SELECT pi.num_training_episodes, el.episode_length
        FROM policy_info pi
        JOIN episode_lengths el ON pi.policy_id = el.policy_id
        ORDER BY pi.num_training_episodes;
        '''
        data = pd.read_sql_query(query, conn)
    return data

def plot_data(data):
    """Plot mean episode lengths with variance against training episodes with stylish design."""
    # Group by training episodes and calculate mean and standard deviation
    grouped_data = data.groupby('num_training_episodes')['episode_length'].agg(['mean', 'std'])

    # Prepare data for plot
    x = grouped_data.index
    y = grouped_data['mean']
    y_upper = y + grouped_data['std']
    y_lower = y - grouped_data['std']

    # Create the plot
    fig = go.Figure()

    # Mean Line (thicker)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Mean Episode Length',
                             line=dict(color='blue', width=4)))

    # Upper Variance
    fig.add_trace(go.Scatter(x=x, y=y_upper, fill=None, mode='lines', line=dict(color='lightblue'),
                             showlegend=False))

    # Lower Variance and fill
    fig.add_trace(go.Scatter(x=x, y=y_lower, fill='tonexty', mode='lines', line=dict(color='lightblue'),
                             showlegend=False))

    # Add titles and labels with a stylish font
    fig.update_layout(
        title="Stylish Plot of Mean Episode Lengths vs. Training Episodes",
        xaxis_title="Training Episodes",
        yaxis_title="Mean Episode Length",
        legend_title="Legend",
        font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(245, 245, 245, 1)'  # Light gray plot background for contrast
    )

    # Add a cool grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')

    fig.show()

def main():
    db_path = 'data/policies.db'  # Replace with your actual database path
    data = fetch_data(db_path)
    plot_data(data)

if __name__ == '__main__':
    main()
