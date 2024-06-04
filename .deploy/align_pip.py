import sys
import argparse
try:
  import importlib.metadata as importlib_metadata # in Python 3.8 and later
except ImportError:
  import importlib_metadata

def get_installed_version(package_name):
  try:
    return importlib_metadata.version(package_name)
  except importlib_metadata.PackageNotFoundError:
    return None

def main(file_path):
  with open(file_path, 'r') as file:
    lines = file.readlines()

  package_versions = {}
  for line in lines:
    line = line.strip()
    if not line:
      continue

    if '==' in line:
      package_name, _ = line.split('==')
    else:
      package_name = line

    installed_version = get_installed_version(package_name)
    if installed_version:
      package_versions[package_name] = f"{package_name}=={installed_version}\n"
    else:
      package_versions[package_name] = f"{package_name} is not installed\n"

  sorted_packages = sorted(package_versions.keys())

  output_lines = [package_versions[package] for package in sorted_packages]

  with open(file_path, 'w') as file:
    file.writelines(output_lines)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Update package versions in a file with the installed versions.')
  parser.add_argument('--file_path', default='.deploy/pip.txt', type=str, help='The path to the file containing package names and versions.')
  args = parser.parse_args()

  main(args.file_path)
