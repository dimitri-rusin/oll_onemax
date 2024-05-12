import argparse
import ast
import hashlib
import io
import itertools
import os
import re
import ruamel.yaml
import socket

def represent_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

def represent_int(dumper, data):
  return dumper.represent_scalar("tag:yaml.org,2002:int", pretty_print_int(data))

def pretty_print_int(n):
  return re.sub(r"(?!^)(?=(?:...)+$)", "_", str(n))

def load_wordlist(filename):
  with open(filename, 'r') as file:
    return [line.strip().split()[1] for line in file]

def generate_filename_from_config(single_config, wordlist):
  # This "default_flow_style=None" makes sure that we get lists with "[]" rather than with "-". Because "-" creates new lines and we want to keep it concise.

  yaml = ruamel.yaml.YAML()
  stream = io.StringIO()
  yaml.dump(single_config, stream)
  config_str = stream.getvalue()

  max_words = 16 # because we have 32 bytes and we use 2 bytes to find one word in the downloaded word list, so 16 words at max
  digest = hashlib.sha256(config_str.encode()).hexdigest()
  words = []
  for i in range(0, max_words * 4, 4):
    index = int(digest[i:i+4], 16) % len(wordlist)
    words.append(wordlist[index])
  return words

def prune_filenames(filenames):
    # Count how many filenames start with each first word
    first_word_count = {words[0]: 0 for words in filenames}
    for words in filenames:
        first_word_count[words[0]] += 1

    pruned_filenames = []
    for words in filenames:
        if first_word_count[words[0]] > 1:
            # If the first word is not unique, use at least two words
            for i in range(2, len(words) + 1):
                candidate_filename = '_'.join(words[:i])
                if candidate_filename not in pruned_filenames:
                    pruned_filenames.append(candidate_filename)
                    break
        else:
            # If the first word is unique, use only one word
            pruned_filenames.append(words[0])

    return pruned_filenames

def safe_eval(expr):
    """
    Evaluate an expression more safely but allow certain types of operations like list and range.
    """
    # Parse the expression to an AST tree
    tree = ast.parse(expr, mode='eval')

    # Define allowed nodes
    allowed_nodes = {
        ast.Expression, ast.Call, ast.Name, ast.BinOp, ast.List, ast.Constant,
        ast.UnaryOp, ast.Load, ast.ListComp, ast.comprehension, ast.Compare,
        ast.Subscript, ast.Slice, ast.Add
    }

    # Define allowed functions and constructors
    allowed_names = {
        'range': range,
        'list': list
    }

    # Function to check nodes recursively
    def check_node(node):
        if type(node) not in allowed_nodes:
            raise ValueError(f"Disallowed expression: {ast.dump(node)}")
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        check_node(item)
            elif isinstance(value, ast.AST):
                check_node(value)

    # Function to evaluate nodes recursively
    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = allowed_names.get(node.func.id, None)
            if func:
                args = [eval_node(arg) for arg in node.args]
                return func(*args)
            else:
                raise ValueError("use of unallowed function: {}".format(node.func.id))
        elif isinstance(node, ast.Name):
            if node.id in allowed_names:
                return allowed_names[node.id]
            raise ValueError(f"Use of unallowed name: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            return left + right
        elif isinstance(node, ast.List):
            return [eval_node(elt) for elt in node.elts]
        elif isinstance(node, ast.Num):
            return node.n
        else:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    # Check and evaluate the AST
    check_node(tree)
    return eval_node(tree.body)

def expand_config(config):
  expanded_config = {}
  for key, value in config.items():
    if isinstance(value, str):
      resulting_list = safe_eval(value)
      assert isinstance(resulting_list, list)
      expanded_config[key] = resulting_list
    elif isinstance(value, dict):
      for subkey, subvalue in value.items():
        expanded_config[f'{key}__{subkey}'] = subvalue
    else:
      expanded_config[key] = value
  return expanded_config

def write_config_to_yaml(configs, wordlist, basename_without_suffix):

  yaml = ruamel.yaml.YAML()
  yaml.indent(mapping=2, sequence=4, offset=2)
  yaml.preserve_quotes = True
  yaml.representer.add_representer(int, represent_int)
  yaml.representer.add_representer(list, represent_list)

  all_filenames = []

  for config in configs:
    expanded_config = expand_config(config)
    keys, values = zip(*expanded_config.items())

    for combination in itertools.product(*values):
      single_config = dict(zip(keys, combination))
      nested_config = {}

      for key, value in single_config.items():
        if '__' in key:
          main_key, sub_key = key.split('__', 1)
          if main_key not in nested_config:
            nested_config[main_key] = {}
          nested_config[main_key][sub_key] = value
        else:
          nested_config[key] = value

      filename_words = generate_filename_from_config(nested_config, wordlist)
      all_filenames.append((nested_config, filename_words))

  pruned_filenames = prune_filenames([words for _, words in all_filenames])

  is_first_iteration = True
  for (single_config, _), wordhash in zip(all_filenames, pruned_filenames):
    hostname = socket.gethostname()
    single_config["db_path"] = f"computed/{hostname}/{basename_without_suffix}/{wordhash}.db"
    stream = io.StringIO()
    yaml.dump(single_config, stream)
    yaml_content = stream.getvalue()

    config_path = f"config/{basename_without_suffix}/{wordhash}.db"
    config_path = config_path.replace("computed", "config")
    config_path = config_path.replace(".db", ".yaml")
    path_parts = os.path.split(config_path)
    directory_path = path_parts[0]
    if is_first_iteration:
      assert not os.path.exists(directory_path), f"Configuration {config_path} already exists. Please run:\nrm -rf {directory_path}"
      os.makedirs(directory_path)
      is_first_iteration = False

    with open(config_path, 'w') as file:
      file.write(yaml_content)

  print(f"All configs have now been generated. To see them, run:\nls -al {directory_path}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Process some YAML.")
  parser.add_argument('file', help='Path to the YAML file to process')
  args = parser.parse_args()
  yaml_file_path = args.file
  yaml = ruamel.yaml.YAML(typ='safe')
  with open(yaml_file_path, 'r') as file:
    configs = yaml.load(file)

  # Download word list from: https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt
  wordlist = load_wordlist('.deploy/eff_large_wordlist.txt')

  basename = os.path.basename(yaml_file_path)
  basename_without_suffix = os.path.splitext(basename)[0]

  write_config_to_yaml(configs, wordlist, basename_without_suffix)
