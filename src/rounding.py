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
      if decision == "smaller":
        smaller_nums = [x for x in action_space if x < value]
        simplified_policy[key] = max(smaller_nums) if smaller_nums else value
      elif decision == "greater":
        greater_nums = [x for x in action_space if x > value]
        simplified_policy[key] = min(greater_nums) if greater_nums else value
  return simplified_policy

def generate_two_simplified_policies(policy, action_space):
  smaller_policy = simplify(policy, action_space, "smaller")
  greater_policy = simplify(policy, action_space, "greater")
  return [('smaller_policy', smaller_policy), ('greater_policy', greater_policy)]

def evaluate(policy, closeness_to_optimum, main_generator):

  dimensionality = len(policy)

  # Compute four parameters to replace the lambda value in every entry.
  parameters_policy = []
  for fitness, lambda_ in policy.items():
    mutation_rate = lambda_ / dimensionality
    crossover_rate = 1.0 / lambda_
    four_parameters = (
      numpy.float64(mutation_rate),
      numpy.int64(lambda_),
      numpy.float64(crossover_rate),
      numpy.int64(lambda_),
    )
    parameters_policy.append(four_parameters)

  episode_seed = main_generator.integers(int(1e9))
  num_function_evaluations, num_evaluation_timesteps = onell_algs_rs.onell_lambda(
    dimensionality,
    parameters_policy,
    episode_seed,
    int(1e9),
    closeness_to_optimum,
  )
  return num_function_evaluations, num_evaluation_timesteps

def statistics(dimensionality, closeness_to_optimum, precision, seed, filepath):

  action_space = [2 ** i for i in range(int(numpy.log2(dimensionality)))]
  theory_derived_size_policy = {
    fitness: int(numpy.sqrt(dimensionality / (dimensionality - fitness))) \
    for fitness in range(dimensionality)
  }
  simplified_policies = generate_two_simplified_policies(theory_derived_size_policy, action_space)
  policies = [('theory_derived_size_policy', theory_derived_size_policy)] + simplified_policies

  main_generator = numpy.random.Generator(numpy.random.MT19937(seed))
  policies_info = []
  for name, policy in policies:
    count = 0
    sum_num_function_evaluations = 0
    sum_num_evaluation_timesteps = 0
    for _ in range(precision):
      num_function_evaluations, num_evaluation_timesteps = evaluate(
        policy = policy,
        closeness_to_optimum = closeness_to_optimum,
        main_generator = main_generator,
      )
      count += 1
      sum_num_function_evaluations += num_function_evaluations
      sum_num_evaluation_timesteps += num_evaluation_timesteps

    avg_num_function_evaluations = sum_num_function_evaluations / count
    avg_num_evaluation_timesteps = sum_num_evaluation_timesteps / count
    policy_info = {
      'seed': seed,
      'closeness_to_optimum': closeness_to_optimum,

      'policy': name,
      'dimensionality': dimensionality,
      'count': count,
      'avg_num_evaluation_timesteps': avg_num_evaluation_timesteps,
      'avg_num_function_evaluations': avg_num_function_evaluations,
    }

    policies_info.append(policy_info)

  with open(filepath, 'w') as json_file:
    print(json.dumps(policies_info, indent = 2), file = json_file)

  return policies_info



policies_info = statistics(
  dimensionality = 500,
  seed = 42,
  precision = 50,
  closeness_to_optimum = 0.5,
  filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'policies_info.json'),
)
