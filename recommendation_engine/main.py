import numpy as np
import random
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# 1. Employee and Data Models
# -----------------------------
class Employee:
    def __init__(self, emp_id, skills, availability, adherence):
        self.emp_id = emp_id
        self.skills = skills          # e.g., {'A': True, 'B': False, 'C': True, 'D': True}
        self.availability = availability  # 1 if available, 0 if not
        self.adherence = adherence    # Value between 0 and 1 (1 means perfectly reliable)
        self.workload = 0             # Hours assigned for the day

    def __repr__(self):
        return (f"Emp({self.emp_id}, workload={self.workload:.1f}, "
                f"adherence={self.adherence:.2f}, avail={self.availability})")

# Daily shift requirements (each skill has a number of required slots)
shift_requirements = {'A': 2, 'B': 4, 'C': 3, 'D': 1}  # Total slots: 10

# -----------------------------
# 2. Ranking Engine Functions
# -----------------------------
def compute_rank(employee, skill, weights):
    """
    Compute a ranking score for an employee for a given skill.
    Skills are binary: 1 if the employee has the skill, else 0.
    Factors: Skill match, workload (to avoid overtime), adherence, availability.
    """
    skill_match = 1 if employee.skills.get(skill, False) else 0
    workload_factor = max(0, 1 - (employee.workload / 8))
    availability = employee.availability
    rank = (weights[0] * skill_match +
            weights[1] * workload_factor +
            weights[2] * employee.adherence +
            weights[3] * availability)
    return rank

def assign_shifts(employees, shift_requirements, weights, shift_length=8):
    """
    Assign employees to each shift slot using the ranking function.
    Ensures that an employee's workload does not exceed 8 hours.
    """
    emp_pool = copy.deepcopy(employees)
    assignments = {}  # {skill: [employee_id, ...]}
    
    for skill, slots in shift_requirements.items():
        assignments[skill] = []
        for _ in range(slots):
            # Filter candidates: available, have required skill, and won't exceed 8 hours.
            candidates = [emp for emp in emp_pool if emp.availability == 1 and 
                          emp.skills.get(skill, False) and (emp.workload + shift_length) <= 8]
            if not candidates:
                assignments[skill].append(None)
                continue
            ranked = sorted(candidates, key=lambda e: compute_rank(e, skill, weights), reverse=True)
            best_candidate = ranked[0]
            best_candidate.workload += shift_length
            assignments[skill].append(best_candidate.emp_id)
    return assignments, emp_pool

# -----------------------------
# 3. Cancellation Simulation
# -----------------------------
def simulate_cancellations(assignments, emp_dict, cancellation_penalty=0.1):
    """
    Simulate cancellation events.
    Each assigned employee cancels with probability = 1 - adherence.
    On cancellation, the employee's adherence is reduced and the slot is set to None.
    """
    cancelled_slots = []  # List of (skill, slot_index)
    for skill in assignments:
        for idx, emp_id in enumerate(assignments[skill]):
            if emp_id is None:
                continue
            employee = emp_dict[emp_id]
            if random.random() < (1 - employee.adherence):
                # Uncomment the following line to print cancellation events:
                # print(f"Employee {emp_id} cancelled for skill {skill} (Adherence before: {employee.adherence:.2f}).")
                employee.adherence = max(employee.adherence - cancellation_penalty, 0)
                assignments[skill][idx] = None
                cancelled_slots.append((skill, idx))
    return assignments, cancelled_slots

def reassign_cancelled_slots(assignments, employees, cancelled_slots, weights, shift_length=8):
    """
    Reassign the cancelled slots using the ranking engine.
    """
    for skill, idx in cancelled_slots:
        candidates = [emp for emp in employees if emp.availability == 1 and 
                      emp.skills.get(skill, False) and (emp.workload + shift_length) <= 8]
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda e: compute_rank(e, skill, weights), reverse=True)
        best_candidate = ranked[0]
        best_candidate.workload += shift_length
        assignments[skill][idx] = best_candidate.emp_id
    return assignments

# -----------------------------
# 4. Global Metrics & Reward Function
# -----------------------------
def evaluate_schedule(assignments, employees, total_slots):
    """
    Evaluate the schedule based on:
      - Coverage rate: fraction of slots filled.
      - Workforce balance: inverse of the standard deviation of workloads.
      - Overtime violations: number of employees working more than 8 hours.
    """
    filled_slots = sum(len([a for a in slot_list if a is not None]) 
                       for slot_list in assignments.values())
    coverage_rate = filled_slots / total_slots
    
    workloads = [emp.workload for emp in employees]
    balance_score = 1 / (np.std(workloads) + 1e-5)
    
    overtime_violations = sum(1 for emp in employees if emp.workload > 8)
    
    return coverage_rate, balance_score, overtime_violations

def compute_reward(coverage_rate, balance_score, overtime_violations, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Compute a reward based on:
      - High coverage (target 100%),
      - High balance score (even workload distribution),
      - Low overtime violations.
    """
    reward = (alpha * coverage_rate) + (beta * balance_score) - (gamma * overtime_violations)
    return reward

# -----------------------------
# 5. Evaluation Function
# -----------------------------
def evaluate_parameters(weights, employees, shift_requirements, shift_length=8, cancellation_penalty=0.1):
    """
    Run a full simulation for one day:
      - Assign shifts.
      - Simulate cancellations.
      - Reassign cancelled slots.
      - Calculate global metrics and reward.
    Returns the reward, final assignments, and updated employees.
    """
    assignments, updated_employees = assign_shifts(employees, shift_requirements, weights, shift_length)
    emp_dict = {emp.emp_id: emp for emp in updated_employees}
    assignments, cancelled_slots = simulate_cancellations(assignments, emp_dict, cancellation_penalty)
    assignments = reassign_cancelled_slots(assignments, updated_employees, cancelled_slots, weights, shift_length)
    
    total_slots = sum(shift_requirements.values())
    coverage_rate, balance_score, overtime_violations = evaluate_schedule(assignments, updated_employees, total_slots)
    reward = compute_reward(coverage_rate, balance_score, overtime_violations)
    return reward, assignments, updated_employees

# -----------------------------
# 6. RL Update Function
# -----------------------------
def rl_update(weights, current_reward, employees, shift_requirements, num_trials=5, 
                       learning_rate=0.05, exploration_rate=0.1, shift_length=8, cancellation_penalty=0.1):
    """
    Enhanced RL update function using more systematic exploration and exploitation.
    Uses gradient-like updates and decreasing exploration over time.
    """
    best_weights = weights.copy()
    best_reward = current_reward
    
    # Try systematic perturbations for each weight dimension
    for i in range(len(weights)):
        # Try increasing and decreasing each weight
        for direction in [-1, 1]:
            new_weights = weights.copy()
            new_weights[i] += direction * learning_rate
            new_weights[i] = max(0, new_weights[i])  # Keep weights non-negative
            
            reward, _, _ = evaluate_parameters(new_weights, employees, 
                                              shift_requirements, shift_length, 
                                              cancellation_penalty)
                                              
            if reward > best_reward:
                best_reward = reward
                best_weights = new_weights.copy()
    
    # Random exploration with controlled rate
    for _ in range(num_trials):
        new_weights = [w + random.uniform(-exploration_rate, exploration_rate) for w in best_weights]
        new_weights = [max(0, w) for w in new_weights]
        
        reward, _, _ = evaluate_parameters(new_weights, employees, 
                                          shift_requirements, shift_length, 
                                          cancellation_penalty)
                                          
        if reward > best_reward:
            best_reward = reward
            best_weights = new_weights.copy()

    return best_weights, best_reward

# -----------------------------
# 7. Employee Roster Creation
# -----------------------------
def create_employees(num_employees):
    """
    Generate a list of employees with:
      - Random binary skills for A, B, C, D.
      - Random availability (70% chance available).
      - Random adherence score between 0.5 and 1.0.
    """
    employees = []
    for i in range(num_employees):
        skills = {s: random.choice([True, False]) for s in ['A', 'B', 'C', 'D']}
        availability = 1 if random.random() < 0.7 else 0
        adherence = random.uniform(0.5, 1.0)
        employees.append(Employee(emp_id=i, skills=skills, availability=availability, adherence=adherence))
    return employees

# -----------------------------
# 8. Training and Validation Function
# -----------------------------
def run_simulation(num_employees, num_episodes, consistent=True):
    """
    Run the training simulation for a given number of employees and episodes.
    If 'consistent' is True, the same employee data is used across episodes.
    Tracks metrics per episode and identifies the best run.
    """
    weights = [1.0, 1.0, 1.0, 1.0]  # Initial ranking weights
    if consistent:
        baseline_employees = create_employees(num_employees)
    
    reward_list = []
    coverage_list = []
    balance_list = []
    overtime_list = []
    
    best_overall_reward = -float("inf")
    best_run_assignments = None
    best_run_weights = None
    best_run_employees = None
    
    for episode in range(num_episodes):
        if consistent:
            employees = copy.deepcopy(baseline_employees)
        else:
            employees = create_employees(num_employees)
        
        current_reward, assignments, updated_employees = evaluate_parameters(weights, employees, shift_requirements)
        total_slots = sum(shift_requirements.values())
        coverage_rate, balance_score, overtime_violations = evaluate_schedule(assignments, updated_employees, total_slots)
        
        reward_list.append(current_reward)
        coverage_list.append(coverage_rate)
        balance_list.append(balance_score)
        overtime_list.append(overtime_violations)
        
        print(f"Episode {episode}: Reward={current_reward:.3f}, Coverage={coverage_rate*100:.1f}%, "
              f"Balance Score={balance_score:.3f}, Overtime Violations={overtime_violations}, Weights={weights}")
        
        if current_reward > best_overall_reward:
            best_overall_reward = current_reward
            best_run_assignments = assignments
            best_run_weights = weights.copy()
            best_run_employees = updated_employees
        
        weights, current_reward = rl_update(weights, current_reward, employees, shift_requirements)
    
    print("\n--- Training Summary ---")
    print(f"Average Reward: {np.mean(reward_list):.3f}")
    print(f"Max Reward: {np.max(reward_list):.3f}")
    print(f"Average Coverage: {np.mean(coverage_list)*100:.1f}%")
    print(f"Average Balance Score: {np.mean(balance_list):.3f}")
    print(f"Average Overtime Violations: {np.mean(overtime_list):.2f}")
    print(f"Final Weights: {weights}")
    
    print("\n--- Best Run ---")
    print(f"Best Overall Reward: {best_overall_reward:.3f}")
    print(f"Best Weights: {best_run_weights}")
    print("Best Run Assignments:")
    for skill, assigned in best_run_assignments.items():
        print(f"  Skill {skill}: {assigned}")
    
    return reward_list, coverage_list, balance_list, overtime_list, weights, best_overall_reward, best_run_assignments, best_run_weights, best_run_employees

# -----------------------------
# 9. Baseline Simulation (Fixed Weights)
# -----------------------------
def run_fixed_simulation(num_employees, num_episodes, fixed_weights=[1.0,1.0,1.0,1.0], consistent=True):
    """
    Run the simulation using fixed weights (no RL updates) to serve as a baseline.
    """
    if consistent:
        baseline_employees = create_employees(num_employees)
    
    reward_list = []
    coverage_list = []
    balance_list = []
    overtime_list = []
    
    for episode in range(num_episodes):
        if consistent:
            employees = copy.deepcopy(baseline_employees)
        else:
            employees = create_employees(num_employees)
        
        reward, assignments, updated_employees = evaluate_parameters(fixed_weights, employees, shift_requirements)
        total_slots = sum(shift_requirements.values())
        coverage, balance, overtime = evaluate_schedule(assignments, updated_employees, total_slots)
        
        reward_list.append(reward)
        coverage_list.append(coverage)
        balance_list.append(balance)
        overtime_list.append(overtime)
    
    return reward_list, coverage_list, balance_list, overtime_list

# -----------------------------
# 10. Sensitivity Analysis Function
# -----------------------------
def sensitivity_analysis(base_weights, employees, shift_requirements, num_trials=10, shift_length=8, cancellation_penalty=0.1):
    """
    Test small perturbations around base_weights and record resulting rewards.
    """
    sensitivity_results = []
    for _ in range(num_trials):
        perturbation = [w + random.uniform(-0.05, 0.05) for w in base_weights]
        perturbation = [max(0, w) for w in perturbation]
        reward, _, _ = evaluate_parameters(perturbation, employees, shift_requirements, shift_length, cancellation_penalty)
        sensitivity_results.append((perturbation, reward))
    return sensitivity_results

# -----------------------------
# 11. Running and Evaluating the Models
# -----------------------------
def rolling_average(data, window_size=5):
    """
    Compute a rolling (moving) average of the data using the specified window size.
    """
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window_data = data[start : i + 1]
        smoothed.append(np.mean(window_data))
    return smoothed

# Run multiple simulations with different episode counts
episode_counts = range(20, 201, 30)  # [20, 50, 80, 110, 140, 170, 200]
all_rl_results = {}
all_baseline_results = {}

for num_episodes in episode_counts:
    print(f"\n=== Running Simulation with {num_episodes} Episodes ===")
    
    # Run RL simulation
    (reward_list_rl, coverage_list_rl, balance_list_rl, overtime_list_rl,
     final_weights, best_reward, best_assignments, best_weights, 
     best_employees) = run_simulation(num_employees=20, num_episodes=num_episodes, consistent=True)
    
    # Run baseline simulation
    baseline_results = run_fixed_simulation(
        num_employees=20, 
        num_episodes=num_episodes, 
        fixed_weights=[1.0,1.0,1.0,1.0], 
        consistent=True
    )
    
    # Store results
    all_rl_results[num_episodes] = rolling_average(reward_list_rl, window_size=5)
    all_baseline_results[num_episodes] = rolling_average(baseline_results[0], window_size=5)
    
    # Create individual plot for each episode count
    plt.figure(figsize=(10, 6))
    plt.plot(all_rl_results[num_episodes], 
             label='RL-Tuned', 
             color='blue')
    plt.plot(all_baseline_results[num_episodes], 
             label='Baseline', 
             color='orange', 
             linestyle='--')
    
    plt.xlabel("Episode")
    plt.ylabel("Reward (Rolling Average)")
    plt.title(f"Reward Trend - {num_episodes} Episodes")
    plt.legend()
    plt.savefig(f"plots/episode_comparison_{num_episodes}.png")
    plt.close()

# Sensitivity Analysis on Final Weights.
print("\n--- Sensitivity Analysis ---")
baseline_employees_sa = create_employees(20)
sensitivity_results = sensitivity_analysis(final_weights, baseline_employees_sa, shift_requirements)
for i, (perturb_weights, perturb_reward) in enumerate(sensitivity_results):
    print(f"Sensitivity Test {i+1}: Weights = {perturb_weights}, Reward = {perturb_reward:.3f}")
