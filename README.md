
![ShiftScheduler drawio](https://github.com/user-attachments/assets/87b83b80-94c5-4196-84c0-1dfe778f10fb)

# Reinforcement Learning for Shift Scheduling

This project implements a reinforcement learning (RL) approach to automate employee shift scheduling. It uses a policy gradient method (REINFORCE) to train a neural network that learns optimal assignment strategies based on employee skills, availability, and workload balancing. The system also simulates real-world scenarios like shift cancellations and uses the trained policy to recommend replacements.

## Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [Usage](#usage)
    * [Running the Simulation](#running-the-simulation)
    * [Customizing the Simulation](#customizing-the-simulation)
* [Project Structure](#project-structure)
* [Data Models](#data-models)
* [Feature Encoding](#feature-encoding)
* [Policy Network](#policy-network)
* [Reinforcement Learning Environment](#reinforcement-learning-environment)
* [Evaluation and Metrics](#evaluation-and-metrics)
* [Shift Cancellation and Replacement](#shift-cancellation-and-replacement)
* [Results and Visualizations](#results-and-visualizations)
* [Future Enhancements](#future-enhancements)
* [Contributing](#contributing)
* [License](#license)

## Overview

The goal of this project is to develop an intelligent system that can efficiently and fairly assign employees to shifts. By using reinforcement learning, the system learns from its past experiences to make better scheduling decisions over time, aiming to maximize shift coverage, balance employee workloads, and ensure that the required skills are matched with the demands of each shift. The project also incorporates a simulation of shift cancellations and a mechanism to find suitable replacements using the trained RL policy.

## Key Features

* **Automated Shift Scheduling:** Uses a trained policy network to assign employees to shifts.
* **Reinforcement Learning:** Employs the REINFORCE algorithm to train the scheduling policy.
* **Skill and Availability Matching:** Ensures that assigned employees possess the required skills and are available for the shift.
* **Workload Balancing:** Encourages a fair distribution of shifts among employees.
* **Shift Cancellation Simulation:** Simulates realistic scenarios where employees may cancel their assigned shifts.
* **Intelligent Replacement Recommendation:** Uses the trained policy to find the best available replacement for cancelled shifts.
* **Performance Evaluation:** Tracks key metrics such as shift coverage, workload standard deviation, and skill match rate.
* **Visualizations:** Generates plots to monitor the training progress and evaluation metrics.
* **Synthetic Data Generation:** Includes functions to create sample employee and shift data for experimentation.
* **Basic Cancellation Detection:** Integrates with a placeholder NLP module to detect cancellations from text messages.

## Getting Started

## Prerequisites
Install requirements.txt using `pip install -r requirements.txt`

## Execution Flow
Generate synthetic employee and shift data.
Initialize the policy network and optimizer.
Train the RL agent for a specified number of episodes.
Create an initial schedule using the trained policy.
Simulate shift cancellations.
Attempt to find replacements for the cancelled shifts.
Print the final results, including coverage and workload distribution.
Save plots of the training progress and evaluation metrics in a plots directory.

To run the default simulation:
`python main.py`

## Project Structure
###The project consists of a single Python file containing the following main components:

#### Global Settings and Constants: Defines lists of days, times, and skills.
Data Models: Classes for Employee and Shift to represent the entities involved in scheduling.
Feature Encoding Functions: Functions to convert Shift and Employee objects into numerical feature vectors.
Policy Network: A PyTorch neural network that learns the scheduling policy.
RL Environment and Training: The ShiftSchedulingEnv class manages the scheduling process, reward calculation, and policy updates.
Additional Visualizations and Metrics: Functions to evaluate the policy and generate plots of training progress.
Shift Cancellation and Replacement: Functions to simulate cancellations and recommend replacements.
Synthetic Data Generation: Functions to create random employee and shift data.
Main Execution Block: Sets up and runs the simulation with specified parameters.

#### Data Models
Employee: Represents an employee with attributes such as id, name, skills, availability, reliability, points, and assigned_shifts.
Shift: Represents a shift with attributes such as id, day, time, required_skill, and assigned_employee.
Feature Encoding
The project uses one-hot encoding to represent categorical features like days, times, and skills. Employee features also include workload, reliability, and points. These encoded features are then used as input to the policy network.

Policy Network
The PolicyNetwork is a simple feedforward neural network with one hidden layer. It takes the concatenated feature vectors of a shift and an employee as input and outputs a scalar score representing the suitability of assigning that employee to the shift.

Reinforcement Learning Environment
The ShiftSchedulingEnv manages the interaction between the RL agent (policy network) and the scheduling task. It handles:

Resetting assignments.
Calculating rewards for assigning employees to shifts based on validity, workload balance, and overtime.
Running episodes where shifts are assigned in a randomized order.
Updating the policy network using the REINFORCE algorithm based on the rewards received.
Evaluation and Metrics
The project evaluates the performance of the trained policy using the following metrics:

Total Reward: The cumulative reward obtained during training episodes.
Shift Coverage: The percentage of shifts that are successfully assigned to an employee.
Workload Standard Deviation: A measure of how evenly the shifts are distributed among employees. A lower standard deviation indicates a more balanced workload.
Average Skill Match: The percentage of assigned shifts where the employee's skills match the required skills.
These metrics are tracked during training and plotted to visualize the learning progress.

Shift Cancellation and Replacement
The simulation includes a basic mechanism for shift cancellations. Employees assigned to shifts might cancel based on messages (detected using the classify_text function). When a cancellation occurs, the system uses the trained policy network to recommend the best available replacement employee who meets the shift requirements.

Results and Visualizations
After training, the script will print the final simulation results, including the final shift coverage and workload distribution. It will also save the following plots in a plots directory:

training_rewards.png: Shows the average reward obtained per evaluation interval during training.
evaluation_metrics.png: Displays plots of evaluation reward, shift coverage, workload standard deviation, and average skill match over the training episodes.

Future Enhancements
More Sophisticated NLP for Cancellation Detection: Integrate a more robust NLP model for accurately detecting cancellations from various message formats.
Consider Employee Preferences: Incorporate employee preferences (e.g., preferred days or times) into the reward function or policy network.
Handle Partial Availability: Allow employees to have more granular availability (e.g., available only in the morning on a specific day).
Dynamic Shift Requirements: Implement scenarios where shift requirements might change dynamically.
Integration with Real-World Data: Adapt the system to work with real employee and shift data from a scheduling platform.
More Advanced RL Algorithms: Explore the use of more advanced RL algorithms like Actor-Critic methods (e.g., A2C, A3C) for potentially faster and more stable learning.
GUI or Web Interface: Develop a user interface to visualize the schedule and allow for manual adjustments.

