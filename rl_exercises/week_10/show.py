import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

log_dir = "logs/"
model_dir = "model/"

env = make_vec_env("LunarLander-v3", n_envs=4)

model = PPO.load(os.path.join(model_dir, "best_model"))
env = gym.make("LunarLander-v3", render_mode="human")

(obs, _) = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _, info = env.step(action)
    if dones:
        (obs, _) = env.reset()
env.close()
