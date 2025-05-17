"""
Evaluation and metrics functions for the recommendation engine.
"""
import numpy as np
import torch
from recommendation_engine.data.feature_encoding import get_feature_vector

def evaluate_policy(env):
    """Evaluate the trained policy"""
    env.reset_assignments()
    total_reward = 0
    
    # Assign shifts with trained policy
    for shift in env.shifts:
        chosen_emp = get_best_employee_for_shift(env, shift)
        reward = assign_if_valid(env, shift, chosen_emp)
        total_reward += reward
    
    # Calculate evaluation metrics
    coverage = calculate_coverage(env.shifts)
    workload_std = calculate_workload_std(env.employees)
    avg_skill_match = calculate_skill_match(env.shifts)
    
    return total_reward, coverage, workload_std, avg_skill_match

def get_best_employee_for_shift(env, shift):
    """Find the best employee for a shift using policy network that has been trained."""
    scores = []
    for emp in env.employees:
        feat = get_feature_vector(shift, emp, len(emp.assigned_shifts))
        feat_tensor = torch.tensor(feat, dtype=torch.float32)
        score = env.policy_net(feat_tensor)
        scores.append(score)
    scores_tensor = torch.stack(scores).squeeze()
    best_employee = torch.argmax(scores_tensor).item()
    return env.employees[best_employee]

def assign_if_valid(env, shift, employee):
    """Assign employee to shift if valid and return reward."""
    reward = env.compute_reward(shift, employee)
    if reward > 0:
        shift.assigned_employee = employee
        employee.assigned_shifts.append(shift)
    return reward

def calculate_coverage(shifts):
    """Calculate percentage of shifts that are assigned."""
    return sum(1 for s in shifts if s.assigned_employee is not None) / len(shifts)

def calculate_workload_std(employees):
    """Calculate standard deviation of employee workloads."""
    workloads = [len(emp.assigned_shifts) for emp in employees]
    return np.std(workloads)

def calculate_skill_match(shifts):
    """Calculate average skill match rate."""
    skill_matches = []
    for s in shifts:
        if s.assigned_employee:
            match = 1.0 if (s.required_skill in s.assigned_employee.skills) else 0.0
            skill_matches.append(match)
    return np.mean(skill_matches) if skill_matches else 0

def track_training_progress(episodes, eval_interval):
    """Initialize data structures to track training progress."""
    return {
        'total_rewards': [],
        'eval_rewards': [],
        'eval_coverages': [],
        'eval_workload_stds': [],
        'eval_skill_matches': []
    }

def evaluate_and_log_progress(env, ep, episodes, eval_interval, metrics):
    """Evaluate current policy and log progress."""
    rwd, cov, wstd, sm = evaluate_policy(env) # Reward, Coverage, Workload STD, Skill Match
    metrics['eval_rewards'].append(rwd)
    metrics['eval_coverages'].append(cov)
    metrics['eval_workload_stds'].append(wstd)
    metrics['eval_skill_matches'].append(sm)
    print(f"Episode {ep+1}/{episodes}: Eval Reward = {rwd:.2f}, Coverage = {cov*100:.1f}%, "
          f"Workload STD = {wstd:.2f}, Skill Match = {sm*100:.1f}%")

def assign_employee_to_shift(shift, employee):
    """Assign employee to shift if they have required skill and availability."""
    is_valid = (shift.required_skill in employee.skills) and employee.is_available(shift.day)
    if is_valid:
        shift.assigned_employee = employee
        employee.assigned_shifts.append(shift)
    return is_valid 