import gymnasium as gym
import environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments setup
config = {"num_dimensions": 30, "random_seed": 2}
vec_env = make_vec_env(lambda: environment.OneMaxOll(config), n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1)
model.save("ppo_OneMaxOll")

# Demonstrating saving and loading
del model
model = PPO.load("ppo_OneMaxOll")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
