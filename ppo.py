from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from tensorboard import program
import environment
import gymnasium as gym
import os

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()
print(f"TensorBoard is running at {url}")

# Configure logger
logger = configure(log_dir, ["stdout", "tensorboard"])

config = {
  'num_dimensions': 10,
  'random_seed': 12345
}
env = make_vec_env(lambda: environment.OneMaxOll(config), n_envs=1)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Callbacks for saving models and evaluations
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="ppo_model")
eval_callback = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=500)

# Train the model
model.learn(total_timesteps=10000, callback=[checkpoint_callback, eval_callback])

# Save the final model
model.save(os.path.join(log_dir, "final_model"))
