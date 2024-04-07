import ast
import inspectify
import os
import sqlite3

def load_config_data(db_path):
    loaded_config = {}
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM CONFIG")
            rows = cursor.fetchall()

        # Process each row to infer the type and construct a nested dictionary
        for key, value in rows:
            # Infer the type
            if value.isdigit():
                parsed_value = int(value)
            elif all(char.isdigit() or char == '.' for char in value):
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value
            elif value.startswith('{') and value.endswith('}'):
                try:
                    # Attempt to parse as a dictionary
                    parsed_value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # If parsing fails, keep the original value
                    parsed_value = value
            else:
                parsed_value = value

            # Create nested dictionaries based on key structure
            key_parts = key.split('__')
            d = loaded_config
            for part in key_parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[key_parts[-1]] = parsed_value
    except sqlite3.Error:
        # If the CONFIG table doesn't exist or any other SQL error occurs
        loaded_config = {}

    return loaded_config

def load_configs_from_folder(folder_path):
    configs = []
    for file in os.listdir(folder_path):
        if file.endswith(".db"):
            db_path = os.path.join(folder_path, file)
            config = load_config_data(db_path)
            configs.append(config)
    return configs

def format_value_for_expression(value):
    if isinstance(value, str):
        # Add quotes around strings
        return f"\"{value}\""
    return value

def match_config_with_filter(config, filter_expr):
    for key, value in filter_expr.items():
        if isinstance(value, dict):
            # Recursive call for nested dictionaries
            if key not in config or not match_config_with_filter(config[key], value):
                return False
        elif isinstance(value, list):
            # Handle lists of expressions
            for i, expr in enumerate(value):
                if i >= len(config.get(key, [])):
                    return False
                config_value = format_value_for_expression(config[key][i])
                expression = expr.replace("{}", str(config_value))
                condition = eval(expression)
                assert type(condition) == bool, f"Filter {key}:{expression} must express a bool."
                if not condition:
                    return False
        else:
            # Handle individual expressions
            config_value = format_value_for_expression(config.get(key))
            expression = value.replace("{}", str(config_value))
            condition = eval(expression)
            assert type(condition) == bool, f"Filter {key}:{expression} must express a bool."
            if not condition:
                return False
    return True

def filter_configs(configs, filter_expr):
    matching_db_paths = []
    for config in configs:
        if match_config_with_filter(config, filter_expr):
            matching_db_paths.append(config.get('db_path'))
    return matching_db_paths



filter_expression = {
  "max_training_timesteps": "{} == {}",
  "ppo": {
    "n_steps": "{} == {}",
    "policy": "{} == {}",
    "batch_size": "{} == 100",
    "gamma": "{} == {}",
    "gae_lambda": "{} <= 0.98",
    "vf_coef": "{} == {}",
    "net_arch": [
      "{} == {}",
      "{} == {}",
    ],
    "learning_rate": "{} == {}",
    "clip_range": "{} == {}",
    "n_epochs": "{} == {}",
    "ent_coef": "{} == {}",
  },
  "n": "{} >= 40",
  "num_timesteps_per_evaluation": "{} == {}",
  "reward_type": "{} == {}",
  "num_evaluation_episodes": "{} == {}",
  "action_type": "{} == {}",
  "num_lambdas": "{} == {}",
  "random_seed": "{} == {}",
  "probability_of_closeness_to_optimum": "{} == {}",
  "state_type": "{} == {}"
}

db_folder_path = '/home/dimitri/code/oll_onemax/computed/cirrus/'



configs = load_configs_from_folder(db_folder_path)
matching_db_paths = filter_configs(configs, filter_expression)

inspectify.d(configs)
inspectify.d(matching_db_paths)
