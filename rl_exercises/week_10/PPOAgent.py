"""
PPO Agent implementation based on RLlib's PPO algorithm.

The generation of this code is supported by the following resources:
- https://docs.ray.io/en/latest/rllib/index.html
- Google Gemini (Search for RLlib functionalities)
- GitHub Copilot (Completions)
"""

import hydra
from omegaconf import DictConfig
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig


@hydra.main(config_path="../configs/agent/", config_name="ppo_HPO", version_base="1.1")
def evaluate(cfg: DictConfig) -> None:
    """
    Evaluate the PPO agent with the given configuration.

    Args:
        cfg (DictConfig): Configuration for the PPO agent.
    """
    # Create a PPO configuration

    """
    Docs:
    https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.html
    https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.evaluation.html
    https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training.html
    https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.core.rl_module.default_model_config.DefaultModelConfig.html
    """
    config = (
        PPOConfig()
        .environment(env=cfg.env_name)
        .framework("torch")
        .rl_module(
            model_config=DefaultModelConfig(
                vf_share_layers=cfg.vf_share_layers,
                fcnet_hiddens=[cfg.hidden_size] * cfg.num_hidden_layers,
                fcnet_activation=cfg.fcnet_activation,
            )
        )
        # .learners(
        #     num_learners=1,
        #     num_gpus_per_learner=1,
        # )
        .training(
            lr=cfg.learning_rate,
            gamma=cfg.gamma,
            train_batch_size_per_learner=cfg.train_batch_size_per_learner,
            num_epochs=cfg.num_epochs,
            use_gae=cfg.use_gae,
            use_critic=cfg.use_critic,
            lambda_=cfg.lambda_,
            vf_loss_coeff=cfg.vf_loss_coeff,
            entropy_coeff=cfg.entropy_coeff,
            clip_param=cfg.clip_param,
            vf_clip_param=cfg.vf_clip_param,
            grad_clip=cfg.grad_clip,
        )
        .evaluation(
            evaluation_interval=cfg.evaluation_interval,
            evaluation_duration=cfg.evaluation_duration,
            evaluation_duration_unit=cfg.evaluation_duration_unit,
        )
    )
    config["seed"] = cfg.seed
    config["num_env_runners"] = 1

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=tune.RunConfig(
            stop={"num_env_steps_sampled_lifetime": cfg.max_timesteps},
        ),
    )

    results = tuner.fit()

    print(
        "Best result:",
        results.get_best_result(metric="episode_reward_mean", mode="max"),
    )


if __name__ == "__main__":
    # Run the evaluation function with the provided configuration
    evaluate()
