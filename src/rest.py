import math
import plotly.graph_objs as go
import sqlite3

db_path = "computed/cirrus/underhand.db"

def load_policy_performance_data(xaxis_choice):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute SQL query based on x-axis choice
    cursor.execute(f'SELECT policy_id, {xaxis_choice} FROM CONSTRUCTED_POLICIES WHERE policy_id >= 1')
    training_data = cursor.fetchall()

    try:
        cursor.execute('SELECT policy_id, num_training_episodes, num_total_function_evaluations, num_total_timesteps FROM CONSTRUCTED_POLICIES WHERE policy_id >= 1')
        rows = cursor.fetchall()
        policy_id_to_x_values = {policy_id: {column_name: value for column_name, value in zip(['num_training_episodes', 'num_total_function_evaluations', 'num_total_timesteps'], row)}
                                 for policy_id, *row in rows}
    except sqlite3.OperationalError:
        cursor.execute('SELECT policy_id, num_training_episodes FROM CONSTRUCTED_POLICIES WHERE policy_id >= 1')
        rows = cursor.fetchall()
        policy_id_to_x_values = {policy_id: {'num_training_episodes': num_training_episodes}
                                 for policy_id, num_training_episodes in rows}

    avg_function_evaluations, std_dev_evaluations = [], []
    for policy_id, _ in training_data:
        cursor.execute('SELECT num_function_evaluations FROM EVALUATION_EPISODES WHERE policy_id = ?', (policy_id,))
        evaluations = [e[0] for e in cursor.fetchall()]
        num_evaluation_episodes = len(evaluations)  # Assuming number of episodes is length of evaluations

        avg_evaluations = sum(evaluations) / len(evaluations) if evaluations else None
        std_dev = math.sqrt(sum((e - avg_evaluations) ** 2 for e in evaluations) / len(evaluations)) if evaluations else 0
        avg_function_evaluations.append(avg_evaluations if avg_evaluations is not None else 0)
        std_dev_evaluations.append(std_dev)

    policy_ids, num_episodes = zip(*training_data) if training_data else ([], [])

    cursor.execute('SELECT AVG(num_function_evaluations) FROM EVALUATION_EPISODES WHERE policy_id = -1')
    baseline_avg_length = 0

    cursor.execute('SELECT num_function_evaluations FROM EVALUATION_EPISODES WHERE policy_id = -1')
    baseline_evaluations = [e[0] for e in cursor.fetchall()]
    baseline_variance = sum((e - baseline_avg_length) ** 2 for e in baseline_evaluations) / (len(baseline_evaluations) - 1) if len(baseline_evaluations) > 1 else 0
    baseline_std_dev = math.sqrt(baseline_variance)

    baseline_upper_bound = [baseline_avg_length + baseline_std_dev] * len(num_episodes)
    baseline_lower_bound = [baseline_avg_length - baseline_std_dev] * len(num_episodes)

    data = [
        go.Scatter(x=num_episodes, y=avg_function_evaluations, mode='lines+markers', name='#FEs until optimum', line=dict(color='blue', width=4)),
        go.Scatter(x=num_episodes, y=[avg + std for avg, std in zip(avg_function_evaluations, std_dev_evaluations)], mode='lines', line=dict(color='rgba(173,216,230,0.2)'), name='Upper Bound (Mean + Std. Dev.)'),
        go.Scatter(x=num_episodes, y=[avg - std for avg, std in zip(avg_function_evaluations, std_dev_evaluations)], mode='lines', fill='tonexty', line=dict(color='rgba(173,216,230,0.2)'), name='Lower Bound (Mean - Std. Dev.)'),
        go.Scatter(x=[min(num_episodes), max(num_episodes)] if num_episodes else [0], y=[baseline_avg_length, baseline_avg_length], mode='lines', name='Theory: √(𝑛/(𝑛 − 𝑓(𝑥)))', line=dict(color='orange', width=2, dash='dash')),
        go.Scatter(x=num_episodes, y=baseline_upper_bound, mode='lines', line=dict(color='rgba(255, 165, 0, 0.2)'), name='Upper Bound (Baseline Variance)'),
        go.Scatter(x=num_episodes, y=baseline_lower_bound, mode='lines', fill='tonexty', line=dict(color='rgba(255, 165, 0, 0.2)'), name='Lower Bound (Baseline Variance)'),
    ]

    conn.close()
    return data

# Choose x-axis (e.g., 'num_training_episodes', 'num_total_function_evaluations', 'num_total_timesteps')
xaxis_choice = 'num_total_timesteps'

data = load_policy_performance_data(xaxis_choice)

layout = go.Layout(
    title='Policy Performance Plot',
    xaxis=dict(title=xaxis_choice.replace('_', ' ').title()),
    yaxis=dict(title='#FEs until optimum'),
    font=dict(family='Courier New, monospace', size=18, color='RebeccaPurple'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(245, 245, 245, 1)'
)

fig = go.Figure(data=data, layout=layout)
fig.show()
