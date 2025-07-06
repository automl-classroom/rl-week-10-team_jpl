import pandas as pd
import matplotlib.pyplot as plt
from ray.tune.analysis import ExperimentAnalysis

experiment_path = "/Users/philipp/Documents/Studium/Informatik/Semester_3/RL/Assignments/rl-week-10-team_jpl/rl_exercises/week_10/logs/PPO_1"

# Load the experiment analysis object
analysis = ExperimentAnalysis(experiment_path)

# Get all trial results as a list of DataFrames
trial_dfs = analysis.trial_dataframes

# Plot episode_reward_mean for each trial
plt.figure(figsize=(10, 6))

for i, df in enumerate(trial_dfs.values()):
    plt.plot(df["training_iteration"], df["episode_reward_mean"], label=f"Trial {i+1}")

plt.xlabel("Training Iterations")
plt.ylabel("Mean Episode Reward")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.show()
