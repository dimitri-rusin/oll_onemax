import yaml
import os
import itertools
import hashlib

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
    return '_'.join(words)

def write_config_to_yaml(directory, config, wordlist, num_words):
    keys, values = zip(*config.items())
    for combination in itertools.product(*values):
        single_config = dict(zip(keys, combination))
        filename = generate_filename_from_config(single_config, wordlist, num_words)
        single_config["db_path"] = single_config["db_path"].replace("<wordhash>", filename)
        yaml_content = yaml.dump(single_config)
        filepath = os.path.join(directory, f'{filename}.yaml')

        with open(filepath, 'w') as file:
            file.write(yaml_content)

# Define the configuration dictionary with lists of possible values
config = {
  "db_path": ["computed/data/<wordhash>.db"],
  "episodes": [999_999_999_999],
  "epsilon": [0.1],
  "evaluation_interval": [10],
  "gamma": [1, 0.99, 0.995, 0.9998],
  "learning_rate": [0.1],
  "n": [50, 500, 1_000],
  "num_evaluation_episodes": [200],
  "num_training_timesteps": [1_000_000],
  "probability_of_closeness_to_optimum": [0.5, 0.9],
  "random_seed": [42]
}

wordlist = load_wordlist('.deploy/eff_large_wordlist.txt')

directory = 'config/March_11'
os.makedirs(directory, exist_ok=True)

write_config_to_yaml(directory, config, wordlist, num_words=5)
