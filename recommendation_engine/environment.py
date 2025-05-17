"""
Reinforcement Learning environment for shift scheduling.
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from recommendation_engine.feature_encoding import get_feature_vector

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