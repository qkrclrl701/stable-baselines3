import os
import pybullet_envs

from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

env = make_vec_env("HumanoidFlagrunBulletEnv-v0", n_envs=1)

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=2000)

# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "/tmp/"
model.save(log_dir + "ppo_humanoid")
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
env.save(stats_path)

# Load the agent
model = PPO.load(log_dir + "ppo_humanoid")

# Load the saved statistics
env = make_vec_env("HumanoidFlagrunBulletEnv-v0", n_envs=1)
env = VecNormalize.load(stats_path, env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")