import os
import sys
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description="Path to an experiment settings file.")
    parser.add_argument('filepath', type=str, help='The path to the file with experiment settings.')
    parser.add_argument('--clean', action='store_true', help='Clean the environment variables instead of setting them')
    args = parser.parse_args()

    # Detect the shell type
    shell = os.getenv('SHELL')

    try:
        with open(args.filepath) as file:
            config = yaml.safe_load(file)
        filename = os.path.basename(args.filepath)
        set_or_unset_env_vars(config, args.clean, shell)
    except FileNotFoundError:
        print(f"Error: '{args.filepath}' does not exist.", file=sys.stderr)
        sys.exit(1)

def set_or_unset_env_vars(config, clean, shell):
    for key, value in flatten_config(config, 'OO'):
        env_var_name = key.upper()
        if clean:
            if 'fish' in shell:
                print(f'set -e {env_var_name}')
            else:
                print(f'unset {env_var_name}')
        else:
            if isinstance(value, str):
                print(f'export {env_var_name}="{value}"')
            else:
                # Remove underscores for numerical values and export
                value_str = str(value).replace('_', '')
                print(f'export {env_var_name}={value_str}')

def flatten_config(config, parent_key=''):
    items = []
    for k, v in config.items():
        new_key = f'{parent_key}_{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key + '_'))
        else:
            items.append((new_key, v))
    return items

if __name__ == "__main__":
    main()
