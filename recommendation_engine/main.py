"""
=============================================================
AI-Powered Workforce Shift Optimization Prototype
=============================================================

Project Objective:
- Automatically assign shifts to employees based on their skills,
  availability, and workload.
- Penalize invalid assignments and encourage workload balance.
- Simulate last-minute cancellations and dynamically assign replacements
  using an ML-based ranking of employees.
- Provide transparency via a point-based reward/penalty system.

Expected Outcomes:
✅ Automated shift scheduling based on employee availability & skill set.
✅ Reduction in last-minute cancellations with penalties for repeat offenders.
✅ Faster shift replacements using an ML-based ranking engine.
✅ Fairness in workforce allocation via a point-based system.

This prototype is designed for an entry-level candidate.
It uses a simple feed-forward policy network (PyTorch) trained with REINFORCE,
synthetic data, and matplotlib (Agg backend) to visualize training progress.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set Agg backend for matplotlib (for headless systems such as WSL)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------
# Global Settings and Constants
# ----------------------------
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["Morning", "Evening"]
SKILLS = ["Customer Service", "Technical Support", "Sales", "Management"]

# ----------------------------
# Data Models
# ----------------------------
class Employee:
    def __init__(self, emp_id, name, skills, availability, reliability=0.9):
        """
        :param skills: list of skills (subset of SKILLS)
        :param availability: list of days employee is available (subset of DAYS)
        """
        self.id = emp_id
        self.name = name
        self.skills = skills
        self.availability = availability  # e.g., ["Monday", "Wednesday", "Friday"]
        self.reliability = reliability    # Probability of showing up
        self.points = 100                 # Starting points
        self.assigned_shifts = []         # List of shifts assigned in current schedule

    def is_available(self, day):
        return day in self.availability

class Shift:
    def __init__(self, shift_id, day, time, required_skill):
        """
        :param day: day of week (from DAYS)
        :param time: time of day (from TIMES)
        :param required_skill: required skill (one from SKILLS)
        """
        self.id = shift_id
        self.day = day
        self.time = time
        self.required_skill = required_skill
        self.assigned_employee = None

# ----------------------------
# Feature Encoding Functions
# ----------------------------
def encode_shift(shift):
    """
    Encode shift into a feature vector.
    Features:
      - Day one-hot (len = len(DAYS))
      - Time one-hot (len = len(TIMES))
      - Required skill one-hot (len = len(SKILLS))
    """
    day_vec = [1 if d == shift.day else 0 for d in DAYS]
    time_vec = [1 if t == shift.time else 0 for t in TIMES]
    skill_vec = [1 if s == shift.required_skill else 0 for s in SKILLS]
    return np.array(day_vec + time_vec + skill_vec, dtype=np.float32)

def encode_employee(emp, current_workload):
    """
    Encode employee into a feature vector.
    Features:
      - Skills one-hot (len = len(SKILLS))
      - Availability one-hot for each day (len = len(DAYS))
      - Current workload (normalized scalar)
      - Reliability (scalar)
      - Points (scalar, normalized by 100)
    """
    skill_vec = [1 if s in emp.skills else 0 for s in SKILLS]
    avail_vec = [1 if d in emp.availability else 0 for d in DAYS]
    workload = np.array([current_workload], dtype=np.float32)
    reliability = np.array([emp.reliability], dtype=np.float32)
    points = np.array([emp.points / 100.0], dtype=np.float32)
    return np.concatenate([np.array(skill_vec, dtype=np.float32),
                           np.array(avail_vec, dtype=np.float32),
                           workload, reliability, points])

def get_feature_vector(shift, emp, emp_workload):
    """
    Concatenate shift and employee features.
    """
    shift_feat = encode_shift(shift)
    emp_feat = encode_employee(emp, emp_workload)
    return np.concatenate([shift_feat, emp_feat])

# ----------------------------
# Policy Network (ML Model)
# ----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim= (len(DAYS)+len(TIMES)+len(SKILLS)) + (len(SKILLS)+len(DAYS)+3), hidden_dim=32):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Outputs a scalar score

    def forward(self, x):
        x = F.relu(self.fc1(x))
        score = self.fc2(x)
        return score

# ----------------------------
# RL Environment and Training
# ----------------------------
class ShiftSchedulingEnv:
    def __init__(self, employees, shifts, policy_net, optimizer):
        self.employees = employees
        self.shifts = shifts
        self.policy_net = policy_net
        self.optimizer = optimizer

    def reset_assignments(self):
        for emp in self.employees:
            emp.assigned_shifts = []
        for sh in self.shifts:
            sh.assigned_employee = None

    def compute_reward(self, shift, emp):
        """
        Compute reward for assigning an employee to a shift.
        Valid assignment: employee has the required skill and is available.
        Reward = 1.0 plus a bonus equal to (average workload - employee workload).
        This bonus is positive if the employee has fewer assignments than average,
        and negative if higher, encouraging balanced assignments.
        Invalid assignment yields -1.0.
        """
        valid = (shift.required_skill in emp.skills) and emp.is_available(shift.day)
        workloads = [len(e.assigned_shifts) for e in self.employees]
        avg_workload = np.mean(workloads) if workloads else 0
        emp_workload = len(emp.assigned_shifts)
        bonus = (avg_workload - emp_workload)  # more aggressive bonus/penalty
        return (1.0 + bonus) if valid else -1.0

    def run_episode(self, training=True):
        """
        Run one episode: assign each shift in randomized order.
        Returns a trajectory of (log_prob, reward) tuples.
        """
        self.reset_assignments()
        trajectory = []
        random_shifts = self.shifts[:]
        random.shuffle(random_shifts)
        for shift in random_shifts:
            scores = []
            for emp in self.employees:
                feat = get_feature_vector(shift, emp, len(emp.assigned_shifts))
                feat_tensor = torch.tensor(feat, dtype=torch.float32)
                score = self.policy_net(feat_tensor)
                scores.append(score)
            scores_tensor = torch.stack(scores).squeeze()
            probs = F.softmax(scores_tensor, dim=0)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            chosen_emp = self.employees[action.item()]
            reward = self.compute_reward(shift, chosen_emp)
            # Only assign if valid (reward > 0)
            if reward > 0:
                shift.assigned_employee = chosen_emp
                chosen_emp.assigned_shifts.append(shift)
            trajectory.append((log_prob, reward))
        return trajectory

    def update_policy(self, trajectory):
        """
        Update the policy network using REINFORCE.
        """
        total_reward = sum(r for (_, r) in trajectory)
        loss = 0
        for log_prob, _ in trajectory:
            loss -= log_prob * total_reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return total_reward

# ----------------------------
# Additional Visualizations and Metrics
# ----------------------------
def evaluate_policy(env):
    """
    Evaluate the trained policy deterministically.
    Returns: total_reward, coverage, workload_std, avg_skill_match.
    Uses the same compute_reward for consistency.
    """
    env.reset_assignments()
    total_reward = 0
    for shift in env.shifts:
        scores = []
        for emp in env.employees:
            feat = get_feature_vector(shift, emp, len(emp.assigned_shifts))
            feat_tensor = torch.tensor(feat, dtype=torch.float32)
            score = env.policy_net(feat_tensor)
            scores.append(score)
        scores_tensor = torch.stack(scores).squeeze()
        best_action = torch.argmax(scores_tensor).item()
        chosen_emp = env.employees[best_action]
        rwd = env.compute_reward(shift, chosen_emp)
        if rwd > 0:
            shift.assigned_employee = chosen_emp
            chosen_emp.assigned_shifts.append(shift)
        total_reward += rwd
    coverage = sum(1 for s in env.shifts if s.assigned_employee is not None) / len(env.shifts)
    workloads = [len(emp.assigned_shifts) for emp in env.employees]
    workload_std = np.std(workloads)
    skill_matches = []
    for s in env.shifts:
        if s.assigned_employee:
            match = 1.0 if (s.required_skill in s.assigned_employee.skills) else 0.0
            skill_matches.append(match)
    avg_skill_match = np.mean(skill_matches) if skill_matches else 0
    return total_reward, coverage, workload_std, avg_skill_match

def train_agent(env, episodes=500, eval_interval=50):
    total_rewards = []
    eval_rewards = []
    eval_coverages = []
    eval_workload_stds = []
    eval_skill_matches = []
    for ep in range(episodes):
        trajectory = env.run_episode(training=True)
        ep_reward = env.update_policy(trajectory)
        total_rewards.append(ep_reward)
        if (ep+1) % eval_interval == 0:
            rwd, cov, wstd, sm = evaluate_policy(env)
            eval_rewards.append(rwd)
            eval_coverages.append(cov)
            eval_workload_stds.append(wstd)
            eval_skill_matches.append(sm)
            print(f"Episode {ep+1}/{episodes}: Eval Reward = {rwd:.2f}, Coverage = {cov*100:.1f}%, Workload STD = {wstd:.2f}, Skill Match = {sm*100:.1f}%")
    # Plot training rewards
    plt.figure(figsize=(8,6))
    plt.plot(total_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Progress")
    plt.legend()
    plt.savefig("training_rewards.png")
    print("Saved training reward plot as 'training_rewards.png'")
    # Plot evaluation metrics
    fig, axs = plt.subplots(2,2, figsize=(12,10))
    episodes_axis = np.arange(eval_interval, episodes+1, eval_interval)
    axs[0,0].plot(episodes_axis, eval_rewards, marker="o")
    axs[0,0].set_title("Evaluation Reward")
    axs[0,0].set_xlabel("Episode")
    axs[0,0].set_ylabel("Reward")
    axs[0,1].plot(episodes_axis, [c*100 for c in eval_coverages], marker="o", color="g")
    axs[0,1].set_title("Shift Coverage (%)")
    axs[0,1].set_xlabel("Episode")
    axs[0,1].set_ylabel("Coverage (%)")
    axs[1,0].plot(episodes_axis, eval_workload_stds, marker="o", color="r")
    axs[1,0].set_title("Workload STD")
    axs[1,0].set_xlabel("Episode")
    axs[1,0].set_ylabel("Std Dev")
    axs[1,1].plot(episodes_axis, [sm*100 for sm in eval_skill_matches], marker="o", color="m")
    axs[1,1].set_title("Average Skill Match (%)")
    axs[1,1].set_xlabel("Episode")
    axs[1,1].set_ylabel("Skill Match (%)")
    plt.tight_layout()
    plt.savefig("evaluation_metrics.png")
    print("Saved evaluation metrics plot as 'evaluation_metrics.png'")
    return total_rewards

# ----------------------------
# ML-Based Replacement Ranking
# ----------------------------
def recommend_replacement(shift, employees, policy_net):
    """
    Use the trained policy network to score and rank employees for a cancelled shift.
    Returns the best candidate among those valid (has required skill and is available).
    """
    candidate_scores = []
    for emp in employees:
        feat = get_feature_vector(shift, emp, len(emp.assigned_shifts))
        feat_tensor = torch.tensor(feat, dtype=torch.float32)
        score = policy_net(feat_tensor).item()
        if (shift.required_skill in emp.skills) and emp.is_available(shift.day):
            candidate_scores.append((emp, score))
    if candidate_scores:
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return candidate_scores[0][0]
    else:
        return None

# ----------------------------
# Synthetic Data Generation
# ----------------------------
def generate_employees(n=5):
    """
    Generate synthetic employees.
    Each employee has a random subset of skills and is available on a random set of days.
    """
    employees = []
    for i in range(n):
        num_skills = random.randint(1, len(SKILLS))
        emp_skills = random.sample(SKILLS, num_skills)
        avail_days = random.sample(DAYS, random.randint(3, len(DAYS)))
        reliability = round(random.uniform(0.7, 1.0), 2)
        employees.append(Employee(i, f"Employee_{i}", emp_skills, avail_days, reliability))
    return employees

def generate_shifts(n=10):
    """
    Generate synthetic shifts.
    Each shift is assigned a random day, time, and required skill.
    """
    shifts = []
    for i in range(n):
        day = random.choice(DAYS)
        time = random.choice(TIMES)
        required_skill = random.choice(SKILLS)
        shifts.append(Shift(i, day, time, required_skill))
    return shifts

# ----------------------------
# Main Flow
# ----------------------------
def run_simulation(n_employees=8, n_shifts=20, episodes=5000, eval_interval=50):
    """
    Run a complete shift scheduling simulation with specified parameters.
    
    Args:
        n_employees: Number of employees to generate
        n_shifts: Number of shifts to generate
        episodes: Number of training episodes
        eval_interval: Interval for evaluation during training
        
    Returns:
        dict: Results including coverage, workloads, and other metrics
    """
    # Generate synthetic employees and shifts
    employees = generate_employees(n=n_employees)
    shifts = generate_shifts(n=n_shifts)

    # Initialize policy network and optimizer
    policy_net = PolicyNetwork()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    # Create RL environment
    env = ShiftSchedulingEnv(employees, shifts, policy_net, optimizer)

    print(f"Training RL policy for {n_employees} employees and {n_shifts} shifts...")
    train_agent(env, episodes=episodes, eval_interval=eval_interval)

    # Use the trained policy to schedule shifts deterministically
    print("\nScheduling shifts using the trained policy...")
    env.reset_assignments()
    for shift in shifts:
        scores = []
        for emp in employees:
            feat = get_feature_vector(shift, emp, len(emp.assigned_shifts))
            feat_tensor = torch.tensor(feat, dtype=torch.float32)
            score = policy_net(feat_tensor)
            scores.append(score)
        scores_tensor = torch.stack(scores).squeeze()
        best_action = torch.argmax(scores_tensor).item()
        chosen_emp = employees[best_action]
        if (shift.required_skill in chosen_emp.skills) and chosen_emp.is_available(shift.day):
            shift.assigned_employee = chosen_emp
            chosen_emp.assigned_shifts.append(shift)
    valid_assignments = sum(1 for s in shifts if s.assigned_employee is not None)
    print(f"Initial Schedule: {valid_assignments}/{len(shifts)} shifts assigned.")

    # Simulate Cancellations
    cancellation_rate = 0.3  # 30% chance for each shift to cancel
    cancelled_shifts = []
    for shift in shifts:
        if shift.assigned_employee and random.random() < cancellation_rate:
            shift.assigned_employee.points -= 10
            shift.assigned_employee.assigned_shifts.remove(shift)
            shift.assigned_employee = None
            cancelled_shifts.append(shift)
    print(f"Simulated cancellations: {len(cancelled_shifts)} shifts cancelled.")

    # Use ML-based ranking to replace cancelled shifts
    replacements = 0
    for shift in cancelled_shifts:
        replacement = recommend_replacement(shift, employees, policy_net)
        if replacement:
            shift.assigned_employee = replacement
            replacement.assigned_shifts.append(shift)
            replacements += 1
    print(f"Replacements found for {replacements} out of {len(cancelled_shifts)} cancelled shifts.")

    # Final Reporting
    final_coverage = sum(1 for s in shifts if s.assigned_employee is not None)
    workloads = [len(emp.assigned_shifts) for emp in employees]
    
    print("\n=== Final Results ===")
    print(f"Total Shifts: {len(shifts)}")
    print(f"Final Coverage: {final_coverage}/{len(shifts)} ({(final_coverage/len(shifts))*100:.1f}%)")
    for emp in employees:
        print(f"{emp.name}: {len(emp.assigned_shifts)} shifts, Points: {emp.points}, Reliability: {emp.reliability}")
    
    # Return results for analysis
    return {
        "n_employees": n_employees,
        "n_shifts": n_shifts,
        "initial_coverage": valid_assignments/len(shifts),
        "final_coverage": final_coverage/len(shifts),
        "cancellations": len(cancelled_shifts),
        "replacements": replacements,
        "workload_std": np.std(workloads),
        "workloads": workloads
    }

if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Test with different combinations of employees and shifts
    results = []
    
    # Test with varying number of employees
    for n_emp in range(5, 6):
        result = run_simulation(n_employees=n_emp, n_shifts=30)
        results.append(result)
    
    # Test with varying number of shifts
    #for n_shifts in [10, 15, 20, 25, 30]:
    #    result = run_simulation(n_employees=8, n_shifts=n_shifts)
    #    results.append(result)
    
    # Compare results
    print("\n=== Comparative Results ===")
    print("Employee Scaling:")
    for r in results[:6]:
        print(f"Employees: {r['n_employees']}, Final Coverage: {r['final_coverage']*100:.1f}%, Workload STD: {r['workload_std']:.2f}")
    
    print("\nShift Scaling:")
    for r in results[6:]:
        print(f"Shifts: {r['n_shifts']}, Final Coverage: {r['final_coverage']*100:.1f}%, Workload STD: {r['workload_std']:.2f}")
