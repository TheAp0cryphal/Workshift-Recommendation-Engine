"""
Main entry point for the recommendation engine.
"""
import random
import numpy as np
import torch
import argparse
import matplotlib
matplotlib.use('Agg')  # Set Agg backend for matplotlib (for WSL)

from recommendation_engine.simulation.simulation import run_simulation

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run shift scheduling simulation')
    parser.add_argument('--employees', type=int, default=5, help='Number of employees')
    parser.add_argument('--shifts', type=int, default=10, help='Number of shifts')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--eval_interval', type=int, default=250, help='Evaluation interval')
    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Run simulation with provided parameters
    result = run_simulation(
        n_employees=args.employees, 
        n_shifts=args.shifts, 
        episodes=args.episodes, 
        eval_interval=args.eval_interval
    )
    
    # Print results
    print("\n=== Simulation Results ===")
    print(f"Employees: {args.employees}, Shifts: {args.shifts}")
    print(f"Final Coverage: {result['final_coverage']*100:.1f}%")
    print(f"Workload STD: {result['workload_std']:.2f}")
