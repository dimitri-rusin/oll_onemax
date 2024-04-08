import argparse
import hashlib
import itertools
import os
import re
import socket
import yaml



def represent_int(dumper, data):
  return dumper.represent_scalar("tag:yaml.org,2002:int", pretty_print_int(data))

def pretty_print_int(n):
  return re.sub(r"(?!^)(?=(?:...)+$)", "_", str(n))

def load_wordlist(filename):
  with open(filename, 'r') as file:
    return [line.strip().split()[1] for line in file]

def generate_filename_from_config(single_config, wordlist, num_words):
  # This "default_flow_style=None" makes sure that we get lists with "[]" rather than with "-". Because "-" creates new lines and we want to keep it concise.
  config_str = yaml.dump(single_config, default_flow_style=None)
  digest = hashlib.sha256(config_str.encode()).hexdigest()
  words = []
  for i in range(0, num_words * 4, 4):
    index = int(digest[i:i+4], 16) % len(wordlist)
    words.append(wordlist[index])
  return words

def prune_filenames(filenames):
  first_word_count = {}
  for words in filenames:
    first_word = words[0]
    first_word_count[first_word] = first_word_count.get(first_word, 0) + 1

  pruned_filenames = []
  for words in filenames:
    if first_word_count[words[0]] == 1:
      pruned_filenames.append(words[0])
    else:
      pruned_filenames.append('_'.join(words))

  return pruned_filenames

def expand_config(config):
  expanded_config = {}
  for key, value in config.items():
    if isinstance(value, dict):
      for subkey, subvalue in value.items():
        expanded_config[f'{key}.{subkey}'] = subvalue
    else:
      expanded_config[key] = value
  return expanded_config

def write_config_to_yaml(configs, wordlist, num_words):
  yaml.add_representer(int, represent_int)
  all_filenames = []

  for config in configs:
    expanded_config = expand_config(config)
    keys, values = zip(*expanded_config.items())

    for combination in itertools.product(*values):
      single_config = dict(zip(keys, combination))
      nested_config = {}

      for key, value in single_config.items():
        if '.' in key:
          main_key, sub_key = key.split('.', 1)
          if main_key not in nested_config:
            nested_config[main_key] = {}
          nested_config[main_key][sub_key] = value
        else:
          nested_config[key] = value

      filename_words = generate_filename_from_config(nested_config, wordlist, num_words)
      all_filenames.append((nested_config, filename_words))

  pruned_filenames = prune_filenames([words for _, words in all_filenames])

  for (single_config, _), pruned_filename in zip(all_filenames, pruned_filenames):
    hostname = socket.gethostname()
    single_config["db_path"] = single_config["db_path"].replace("{wordhash}", pruned_filename)
    config_path = single_config["db_path"]
    single_config["db_path"] = single_config["db_path"].replace("{hostname}", hostname)
    yaml_content = yaml.dump(single_config, default_flow_style=None)


    config_path = config_path.replace("{hostname}/", "")
    config_path = config_path.replace("computed", "config")
    config_path = config_path.replace(".db", ".yaml")
    path_parts = os.path.split(config_path)
    directory_path = path_parts[0]
    os.makedirs(directory_path, exist_ok=True)
    assert not os.path.exists(config_path), f"Configuration {config_path} already exists. Please delete first."
    with open(config_path, 'w') as file:
      file.write(yaml_content)



if __name__ == '__main__':
  yaml_file_path = '.deploy/range.yaml'
  with open(yaml_file_path, 'r') as file:
    configs = yaml.safe_load(file)

  # Download word list from: https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt
  wordlist = load_wordlist('.deploy/eff_large_wordlist.txt')

  write_config_to_yaml(configs, wordlist, num_words=5)
