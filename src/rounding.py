import json
import numpy
import onell_algs_rs
import os

def simplify(policy, action_space, decision):
  simplified_policy = {}
  for key, value in policy.items():
    if value in action_space:
      simplified_policy[key] = value
    else:
      if decision == "rounded_down_policy":
        smaller_nums = [x for x in action_space if x < value]
        simplified_policy[key] = max(smaller_nums) if smaller_nums else value
      elif decision == "rounded_up_policy":
        greater_nums = [x for x in action_space if x > value]
        simplified_policy[key] = min(greater_nums) if greater_nums else value
  return simplified_policy

def generate_two_simplified_policies(policy, action_space):
  rounded_down_policy = simplify(policy, action_space, "rounded_down_policy")
  rounded_up_policy = simplify(policy, action_space, "rounded_up_policy")
  return [('rounded_down_policy', rounded_down_policy), ('rounded_up_policy', rounded_up_policy)]

def evaluate(policy, closeness_to_optimum, episode_seed):

  dimensionality = len(policy)

  # Compute four parameters to replace the lambda value in every entry.
  parameters_policy = []
  for fitness, lambda_ in policy.items():
    mutation_rate = lambda_ / dimensionality
    crossover_rate = 1.0 / lambda_
    four_parameters = (
      numpy.float64(mutation_rate),
      numpy.int64(numpy.around(lambda_)),
      numpy.float64(crossover_rate),
      numpy.int64(numpy.around(lambda_)),
    )
    parameters_policy.append(four_parameters)

  num_function_evaluations, num_evaluation_timesteps = onell_algs_rs.onell_lambda(
    n = dimensionality,
    oll_parameters = parameters_policy,
    seed = episode_seed,
    max_evals = int(1e9),
    probability = closeness_to_optimum,
  )
  return num_function_evaluations, num_evaluation_timesteps

def custom_converter(obj):
  if 'details' in obj and not type(obj['details']) == str:
    # Convert details dictionary to a one-line string
    details_str = json.dumps(list(obj['details']), separators=(', ', ':'))
    obj['details'] = details_str
  return obj

def statistics(dimensionalities, closeness_to_optimum, precision, seed, filepath):

  num_files = 0
  policies_info = []
  for dimensionality in dimensionalities:
    action_space = [2 ** i for i in range(int(numpy.log2(dimensionality)))]
    theory_derived_policy = {
      fitness: numpy.sqrt(dimensionality / (dimensionality - fitness)) \
      for fitness in range(dimensionality)
    }
    simplified_policies = generate_two_simplified_policies(theory_derived_policy, action_space)
    policies = [('theory_derived_policy', theory_derived_policy)] + simplified_policies

    for name, policy in policies:
      count = 0
      sum_num_function_evaluations = 0
      sum_num_evaluation_timesteps = 0
      sum_squares_evaluations = 0
      sum_squares_timesteps = 0

      main_generator = numpy.random.Generator(numpy.random.MT19937(seed))
      for _ in range(precision):
        episode_seed = main_generator.integers(int(1e9))
        num_function_evaluations, num_evaluation_timesteps = evaluate(
          policy = policy,
          closeness_to_optimum = closeness_to_optimum,
          episode_seed = episode_seed,
        )
        count += 1
        sum_num_function_evaluations += num_function_evaluations
        sum_num_evaluation_timesteps += num_evaluation_timesteps
        sum_squares_evaluations += num_function_evaluations ** 2
        sum_squares_timesteps += num_evaluation_timesteps ** 2

      avg_num_function_evaluations = sum_num_function_evaluations / count
      avg_num_evaluation_timesteps = sum_num_evaluation_timesteps / count
      std_dev_evaluations = numpy.sqrt(sum_squares_evaluations / count - avg_num_function_evaluations ** 2)
      std_dev_timesteps = numpy.sqrt(sum_squares_timesteps / count - avg_num_evaluation_timesteps ** 2)

      policy_info = {
        'dimensionality': dimensionality,
        'policy': name,

        'seed': seed,
        'count': count,
        'avg_num_evaluation_timesteps': avg_num_evaluation_timesteps,
        'std_dev_evaluation_timesteps': std_dev_timesteps,
        'avg_num_function_evaluations': avg_num_function_evaluations,
        'std_dev_function_evaluations': std_dev_evaluations,
        'closeness_to_optimum': closeness_to_optimum,

        'details': policy,
      }

      policies_info.append(policy_info)

      converted_policies = [custom_converter(policy) for policy in policies_info]

      num_files += 1
      with open(f"{filepath}_{num_files}.json", 'w') as json_file:
        json.dump(converted_policies, json_file, indent=2)

  return policies_info



policies_info = statistics(
  dimensionalities = [500, 1_000, 2_000, 3_000],
  seed = 42,
  precision = 10,
  closeness_to_optimum = 0.5,
  filepath = 'computed/policies_info',
)
