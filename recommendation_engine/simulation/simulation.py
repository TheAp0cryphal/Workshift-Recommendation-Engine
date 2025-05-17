"""
Functions for running simulations with the recommendation engine.
"""
import numpy as np
import torch
import torch.optim as optim
import sys
import select

from recommendation_engine.data.data_generator import generate_employees, generate_shifts
from recommendation_engine.models.policy_network import PolicyNetwork
from recommendation_engine.core.environment import ShiftSchedulingEnv
from recommendation_engine.evaluation.evaluation import evaluate_policy, get_best_employee_for_shift, assign_employee_to_shift, track_training_progress, evaluate_and_log_progress
from recommendation_engine.visualization.visualization import plot_training_metrics
from recommendation_engine.data.feature_encoding import get_feature_vector
from recommendation_engine.utils.nlp_module import classify_text

def train_agent(env, episodes=500, eval_interval=50):
    """Train the policy network."""
    metrics = track_training_progress(episodes, eval_interval)
    
    for ep in range(episodes):
        trajectory = env.run_episode()
        ep_reward = env.update_policy(trajectory)
        metrics['total_rewards'].append(ep_reward)
        
        # Check if user pressed Enter to stop training
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            input()  # Clear the input buffer
            print(f"Training stopped early at episode {ep+1}")
            break
            
        if (ep+1) % eval_interval == 0:
            evaluate_and_log_progress(env, ep, episodes, eval_interval, metrics)
    
    plot_training_metrics(metrics, episodes, eval_interval)
    return metrics['total_rewards']

def test_trained_policy(env):
    """Create initial schedule using trained policy."""
    valid_assignments = 0
    for shift in env.shifts:
        chosen_emp = get_best_employee_for_shift(env, shift)
        if assign_employee_to_shift(shift, chosen_emp):
            valid_assignments += 1
    return valid_assignments

def detect_cancellation(message):
    """Detect if a message indicates shift cancellation"""
    return not classify_text(message)

def simulate_cancellations(env):
    """Simulate shift cancellations based on employee messages."""
    cancellation_messages = {
        0: "I cannot make it today", # Cancellation 
        1: "I will not be able to make it today", # Cancellation 
        2: "All good, I'll be there", # No Cancellation
        3: "Due to unforeseen circumstances, I am cancelling my shift", # Cancellation
        4: "I will be coming during my shift", # No Cancellation
        5: "I have a meeting, so I won't be available", # Cancellation
    }
    
    cancelled_shifts = []
    for shift in env.shifts:
        if not shift.assigned_employee:
            continue
            
        # If there's a message for this shift, check if it indicates cancellation
        message = cancellation_messages.get(shift.id, "I'm good")
            
        if detect_cancellation(message):
            print(f"Shift {shift.id} cancellation detected with message: '{message}'")
            shift.assigned_employee.points -= 10
            shift.assigned_employee.assigned_shifts.remove(shift)
            shift.assigned_employee = None
            cancelled_shifts.append(shift)
    
    return cancelled_shifts

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

def handle_cancelled_shifts(env, cancelled_shifts):
    """Find replacements for cancelled shifts."""
    replacements = 0
    for shift in cancelled_shifts:
        replacement = recommend_replacement(shift, env.employees, env.policy_net)
        if replacement:
            shift.assigned_employee = replacement
            replacement.assigned_shifts.append(shift)
            replacements += 1
    return replacements

def generate_simulation_results(env, valid_assignments, cancelled_shifts, replacements):
    """Generate comprehensive results from simulation."""
    final_coverage = sum(1 for s in env.shifts if s.assigned_employee is not None)
    workloads = [len(emp.assigned_shifts) for emp in env.employees]
    
    return {
        "n_employees": len(env.employees),
        "n_shifts": len(env.shifts),
        "initial_coverage": valid_assignments/len(env.shifts),
        "final_coverage": final_coverage/len(env.shifts),
        "cancellations": len(cancelled_shifts),
        "replacements": replacements,
        "workload_std": np.std(workloads),
        "workloads": workloads
    }

def print_final_results(employees, shifts, results):
    """Print final simulation results."""
    final_coverage = int(results["final_coverage"] * len(shifts))
    print("\n=== Final Results ===")
    print(f"Total Shifts: {len(shifts)}")
    print(f"Final Coverage: {final_coverage}/{len(shifts)} ({results['final_coverage']*100:.1f}%)")
    for emp in employees:
        print(f"{emp.name}: {len(emp.assigned_shifts)} shifts, Points: {emp.points}, "
              f"Reliability: {emp.reliability}")

def run_simulation(n_employees=8, n_shifts=20, episodes=500, eval_interval=50):
    """Run a complete shift scheduling simulation with specified parameters."""
    # Generate data and initialize components
    employees = generate_employees(n=n_employees)
    shifts = generate_shifts(n=n_shifts)
    policy_net = PolicyNetwork()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    
    # Initialize environment
    env = ShiftSchedulingEnv(employees, shifts, policy_net, optimizer)

    # Train policy
    print(f"Training RL policy for {n_employees} employees and {n_shifts} shifts...")
    train_agent(env, episodes=episodes, eval_interval=eval_interval)

    # Create initial schedule
    print("\nScheduling shifts using the trained policy...")
    # Reset assignments to ensure no shifts are assigned before creating initial schedule
    env.reset_assignments() 
    # Test the trained policy on how well it schedules shifts

    valid_assignments = test_trained_policy(env)
    print(f"Initial Schedule: {valid_assignments}/{len(shifts)} shifts assigned.")

    # Simulate cancellations
    cancelled_shifts = simulate_cancellations(env)
    print(f"Simulated cancellations: {len(cancelled_shifts)} shifts cancelled.")

    # Handle replacements
    replacements = handle_cancelled_shifts(env, cancelled_shifts)
    print(f"Replacements found for {replacements} out of {len(cancelled_shifts)} cancelled shifts.")

    # Report results
    results = generate_simulation_results(env, valid_assignments, cancelled_shifts, replacements)
    print_final_results(env.employees, env.shifts, results)
    
    return results 