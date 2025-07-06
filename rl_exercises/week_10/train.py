# This code got significant contributions from google gemini 2.5 pro, chatgpt

import os

import gymnasium as gym
import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import torch

SEED = 12


def train_with_config(cfg: DictConfig) -> float:

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    torch.manual_seed(SEED)

    env = make_vec_env(
        cfg.env_id, n_envs=cfg.n_envs, env_kwargs={"gravity": cfg.ppo.training_gravity}
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=cfg.model_dir,
        log_path=cfg.log_dir,
        eval_freq=500,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=cfg.log_dir,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
    )
    model.learn(total_timesteps=cfg.total_timesteps, callback=eval_callback)

    evaluation_gravities = [-9, -10, -11, -11.99]

    eval_rewards: list[float] = []
    for g in evaluation_gravities:

        eval_env = make_vec_env(
            cfg.env_id, n_envs=cfg.n_envs, env_kwargs={"gravity": g}
        )
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
        eval_rewards.append(mean_reward)
        eval_env.close()

    env.close()
    del eval_callback
    del model
    return sum(eval_rewards) / len(eval_rewards)


@hydra.main(config_path="config", config_name="sweep", version_base="1.3")
def main(cfg: DictConfig):
    def objective(trial: optuna.Trial) -> float:
        cfg.ppo.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        cfg.ppo.clip_range = trial.suggest_uniform("clip_range", 0.001, 1.0)
        cfg.ppo.gae_lambda = trial.suggest_uniform("gae_lambda", 0.5, 0.999)
        cfg.ppo.gamma = trial.suggest_uniform("gamma", 0.5, 0.999)
        cfg.ppo.training_gravity = trial.suggest_uniform("training_gravity", -8, -0.001)

        score = train_with_config(cfg)
        return score

    study = optuna.create_study(
        study_name="ppo_hpo_lunarlander",
        storage="sqlite:///ppo_hpo.db",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(study.best_trial)


if __name__ == "__main__":
    main()
