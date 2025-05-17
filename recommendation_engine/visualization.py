"""
Visualization functions for the recommendation engine.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_training_metrics(metrics, episodes, eval_interval):
    """Plot and save training metrics."""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plot_training_rewards(metrics['total_rewards'], episodes, eval_interval)
    plot_evaluation_metrics(metrics, episodes, eval_interval)

def plot_training_rewards(total_rewards, episodes, eval_interval):
    """Plot and save training rewards chart."""
    plt.figure(figsize=(8,6))
    avg_rewards = calculate_average_rewards(total_rewards, eval_interval)
    # Ensure x and y arrays have matching lengths
    num_points = len(avg_rewards)
    episodes_axis = np.arange(eval_interval, eval_interval * (num_points + 1), eval_interval)[:num_points]
    plt.plot(episodes_axis, avg_rewards, label="Avg Reward per 50 Episodes", marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Training Reward Progress (50-Episode Average)")
    plt.legend()
    plt.savefig("plots/training_rewards.png")
    print("Saved training reward plot as 'training_rewards.png'")

def calculate_average_rewards(total_rewards, eval_interval):
    """Calculate average rewards over evaluation intervals."""
    avg_rewards = []
    for i in range(0, len(total_rewards), eval_interval):
        chunk = total_rewards[i:i+eval_interval]
        avg_rewards.append(np.mean(chunk))
    return avg_rewards

def plot_evaluation_metrics(metrics, episodes, eval_interval):
    """Plot and save evaluation metrics charts."""
    fig, axs = plt.subplots(2,2, figsize=(12,10))
    
    # Get actual number of evaluation points
    num_points = len(metrics['eval_rewards'])
    episodes_axis = np.arange(eval_interval, eval_interval * (num_points + 1), eval_interval)[:num_points]
    
    axs[0,0].plot(episodes_axis, metrics['eval_rewards'], marker="o")
    axs[0,0].set_title("Evaluation Reward")
    axs[0,0].set_xlabel("Episode")
    axs[0,0].set_ylabel("Reward")
    
    axs[0,1].plot(episodes_axis, [c*100 for c in metrics['eval_coverages']], marker="o", color="g")
    axs[0,1].set_title("Shift Coverage (%)")
    axs[0,1].set_xlabel("Episode")
    axs[0,1].set_ylabel("Coverage (%)")
    
    axs[1,0].plot(episodes_axis, metrics['eval_workload_stds'], marker="o", color="r")
    axs[1,0].set_title("Workload STD")
    axs[1,0].set_xlabel("Episode")
    axs[1,0].set_ylabel("Std Dev")
    
    axs[1,1].plot(episodes_axis, [sm*100 for sm in metrics['eval_skill_matches']], marker="o", color="m")
    axs[1,1].set_title("Average Skill Match (%)")
    axs[1,1].set_xlabel("Episode")
    axs[1,1].set_ylabel("Skill Match (%)")
    
    plt.tight_layout()
    plt.savefig("plots/evaluation_metrics.png")
    print("Saved evaluation metrics plot as 'evaluation_metrics.png'")
    print("--------------------------------------------------------") 