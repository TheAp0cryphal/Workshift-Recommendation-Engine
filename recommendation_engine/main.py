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
        Overtime (more than one shift per day) is penalized.
        """
        valid = (shift.required_skill in emp.skills) and emp.is_available(shift.day)
        workloads = [len(e.assigned_shifts) for e in self.employees]
        avg_workload = np.mean(workloads) if workloads else 0
        emp_workload = len(emp.assigned_shifts)
        bonus = (avg_workload - emp_workload)  # more aggressive bonus/penalty
        
        # Check for overtime (more than one shift on the same day)
        overtime_penalty = 0
        for assigned_shift in emp.assigned_shifts:
            if assigned_shift.day == shift.day:
                overtime_penalty = -0.5  # Penalty for overtime
                break
            
        return (1.0 + bonus + overtime_penalty) if valid else -1.0

    def run_episode(self):
        """
        Run one episode: assign each shift in randomized order.
        Returns a trajectory of (log_prob, reward) tuples.
        """
        self.reset_assignments()
        trajectory = []
        random_shifts = self.prepare_random_shifts()
        
        for shift in random_shifts:
            # Calculate scores for all employees for this shift using policy network
            scores_tensor = self.score_employees_for_shift(shift)
            
            # Sample an employee (action) based on softmax probability distribution
            # Returns the chosen employee index and log probability for policy gradient
            action, log_prob = self.sample_action(scores_tensor)
            
            # Get the actual employee object from the selected index
            chosen_emp = self.employees[action.item()]
            
            # Calculate reward for this assignment based on skill match, availability and workload
            reward = self.compute_reward(shift, chosen_emp)
            
            # Only assign the shift if the reward is positive (valid assignment)
            self.possibly_assign_shift(shift, chosen_emp, reward)
            
            # Store (log_prob, reward) tuple for policy gradient updates
            trajectory.append((log_prob, reward))
        return trajectory
    
    def prepare_random_shifts(self):
        """Prepare randomized shifts for an episode."""
        random_shifts = self.shifts[:]
        random.shuffle(random_shifts)
        return random_shifts
    
    def score_employees_for_shift(self, shift):
        """Score all employees for a given shift using policy network."""
        scores = []
        for emp in self.employees:
            feat = get_feature_vector(shift, emp, len(emp.assigned_shifts))
            feat_tensor = torch.tensor(feat, dtype=torch.float32)
            score = self.policy_net(feat_tensor)
            scores.append(score)
        return torch.stack(scores).squeeze()
    
    def sample_action(self, scores_tensor):
        """Sample an action from a categorical distribution of scores."""
        probs = F.softmax(scores_tensor, dim=0)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action, log_prob
    
    def possibly_assign_shift(self, shift, employee, reward):
        """Assign shift to employee if reward is positive."""
        if reward > 0:
            shift.assigned_employee = employee
            employee.assigned_shifts.append(shift)

    def update_policy(self, trajectory):
        """
        Update the policy network using REINFORCE.
        """
        # Reward is cumulative reward received after taking an action
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

def train_agent(env, episodes=500, eval_interval=50):
    """Train the policy network."""
    metrics = track_training_progress(episodes, eval_interval)
    
    for ep in range(episodes):
        trajectory = env.run_episode()
        ep_reward = env.update_policy(trajectory)
        metrics['total_rewards'].append(ep_reward)
        
        if (ep+1) % eval_interval == 0:
            evaluate_and_log_progress(env, ep, episodes, eval_interval, metrics)
    
    plot_training_metrics(metrics, episodes, eval_interval)
    return metrics['total_rewards']

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

def plot_training_metrics(metrics, episodes, eval_interval):
    """Plot and save training metrics."""
    plot_training_rewards(metrics['total_rewards'], episodes, eval_interval)
    plot_evaluation_metrics(metrics, episodes, eval_interval)

def plot_training_rewards(total_rewards, episodes, eval_interval):
    """Plot and save training rewards chart."""
    plt.figure(figsize=(8,6))
    avg_rewards = calculate_average_rewards(total_rewards, eval_interval)
    episodes_axis = np.arange(eval_interval, episodes+1, eval_interval)
    plt.plot(episodes_axis, avg_rewards, label="Avg Reward per 50 Episodes", marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Training Reward Progress (50-Episode Average)")
    plt.legend()
    plt.savefig("training_rewards.png")
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
    episodes_axis = np.arange(eval_interval, episodes+1, eval_interval)
    
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
    plt.savefig("evaluation_metrics.png")
    print("Saved evaluation metrics plot as 'evaluation_metrics.png'")
    print("--------------------------------------------------------")

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
    cancelled_shifts = simulate_cancellations(env, cancellation_rate=0.3)
    print(f"Simulated cancellations: {len(cancelled_shifts)} shifts cancelled.")

    # Handle replacements
    replacements = handle_cancelled_shifts(env, cancelled_shifts)
    print(f"Replacements found for {replacements} out of {len(cancelled_shifts)} cancelled shifts.")

    # Report results
    results = generate_simulation_results(env, valid_assignments, cancelled_shifts, replacements)
    print_final_results(env.employees, env.shifts, results)
    
    return results

def test_trained_policy(env):
    """Create initial schedule using trained policy."""
    valid_assignments = 0
    for shift in env.shifts:
        chosen_emp = get_best_employee_for_shift(env, shift)
        if assign_employee_to_shift(shift, chosen_emp):
            valid_assignments += 1
    return valid_assignments

def assign_employee_to_shift(shift, employee):
    """Assign employee to shift if they have required skill and availability."""
    is_valid = (shift.required_skill in employee.skills) and employee.is_available(shift.day)
    if is_valid:
        shift.assigned_employee = employee
        employee.assigned_shifts.append(shift)
    return is_valid

def simulate_cancellations(env, cancellation_rate=0.3):
    """Simulate employee cancellations with specified rate."""
    cancelled_shifts = []
    for shift in env.shifts:
        if shift.assigned_employee and random.random() < cancellation_rate:
            employee = shift.assigned_employee
            handle_cancellation(shift, employee)
            cancelled_shifts.append(shift)
    return cancelled_shifts

def handle_cancellation(shift, employee):
    """Process an employee cancellation."""
    employee.points -= 10
    employee.assigned_shifts.remove(shift)
    shift.assigned_employee = None

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

if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Test with different combinations of employees and shifts
    results = []
    
    # Test with varying number of employees
    for n_emp in range(8, 9):
        result = run_simulation(n_employees=n_emp, n_shifts=30, episodes=20000, eval_interval=500)
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
