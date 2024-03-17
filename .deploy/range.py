import yaml
import os
import itertools
import hashlib
import re

def represent_int(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:int", pretty_print_int(data))

def pretty_print_int(n):
    return re.sub(r"(?!^)(?=(?:...)+$)", "_", str(n))

yaml.add_representer(int, represent_int)

def load_wordlist(filename):
    with open(filename, 'r') as file:
        return [line.strip().split()[1] for line in file]

def generate_filename_from_config(single_config, wordlist, num_words):
    config_str = yaml.dump(single_config)
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

def write_config_to_yaml(directory, configs, wordlist, num_words):
    all_filenames = []
    for config in configs:
        keys, values = zip(*config.items())
        for combination in itertools.product(*values):
            single_config = dict(zip(keys, combination))
            filename_words = generate_filename_from_config(single_config, wordlist, num_words)
            all_filenames.append((single_config, filename_words))

    pruned_filenames = prune_filenames([words for _, words in all_filenames])

    for (single_config, _), pruned_filename in zip(all_filenames, pruned_filenames):
        single_config["db_path"] = single_config["db_path"].replace("<wordhash>", pruned_filename)
        yaml_content = yaml.dump(single_config, default_flow_style=False)
        filepath = os.path.join(directory, f'{pruned_filename}.yaml')

        with open(filepath, 'w') as file:
            file.write(yaml_content)

# Define the configurations list with possible values
configs = [
  {
    "db_path": ["computed/data/<wordhash>.db"],
    "epsilon": [0.2, 0.1, 0.01],
    "evaluation_interval": [500],
    "gamma": [1, 0.99, 0.995, 0.9998],
    "learning_rate": [0.1, 0.2],
    "n": [50],
    "num_evaluation_episodes": [600],
    "max_training_timesteps": [2_000_000],
    "probability_of_closeness_to_optimum": [0.5, 0.9],
    "random_seed": [42]
  },
  {
    "db_path": ["computed/data/<wordhash>.db"],
    "epsilon": [0.1, 0.01],
    "evaluation_interval": [10],
    "gamma": [1, 0.99, 0.995, 0.9998],
    "learning_rate": [0.1],
    "n": [500, 1_000, 2_000],
    "num_evaluation_episodes": [25],
    "max_training_timesteps": [1_000_000],
    "probability_of_closeness_to_optimum": [0.5, 0.9],
    "random_seed": [42]
  },
]

# Download from: https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt
wordlist = load_wordlist('.deploy/eff_large_wordlist.txt')
directory = 'config/a_few_more_configs_March_17'
os.makedirs(directory, exist_ok=True)
write_config_to_yaml(directory, configs, wordlist, num_words=5)
