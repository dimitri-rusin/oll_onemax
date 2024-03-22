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
    for key, value in flatten_config(config, 'OO_'):
        env_var_name = key.upper()
        if clean:
            if 'fish' in shell:
                print(f'set -e {env_var_name}')
            else:
                print(f'unset {env_var_name}')
        else:
            # Convert the value to a string
            value_str = str(value)

            # For Fish shell, use `set -x`
            if 'fish' in shell:
                if isinstance(value, str) or isinstance(value, list):
                    # Use single quotes for string values in Fish shell
                    print(f'set -x {env_var_name} \'{value_str}\'')
                else:
                    # Remove underscores for numerical values
                    value_str = value_str.replace('_', '')
                    print(f'set -x {env_var_name} {value_str}')
            else:
                # For Bash and other shells, use `export`
                if isinstance(value, str) or isinstance(value, list):
                    # Use double quotes for string values in Bash
                    print(f'export {env_var_name}=\"{value_str}\"')
                else:
                    # Remove underscores for numerical values
                    value_str = value_str.replace('_', '')
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
