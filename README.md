
![ShiftScheduler drawio](https://github.com/user-attachments/assets/87b83b80-94c5-4196-84c0-1dfe778f10fb)

Shift Scheduling with Reinforcement Learning
This project implements a reinforcement learning approach to automate shift scheduling for employees. The system uses a policy network to assign shifts based on factors such as skill match, availability, workload balance, and employee reliability.

Features
Employee and Shift Models:
Defines classes to represent employees and shifts with properties like skills, availability, and required shift skills.

Feature Encoding:
Encodes shift and employee data into feature vectors for the policy network.

Policy Network:
A neural network built with PyTorch that scores potential employee-shift assignments.

Reinforcement Learning Environment:
Implements the REINFORCE algorithm to train the policy network. The environment calculates rewards based on valid assignments, workload balancing, and penalties for overtime or cancellations.

Simulation and Evaluation:
Simulates the scheduling process, evaluates performance using metrics (shift coverage, workload standard deviation, and skill match), and generates visualizations of training progress.

Cancellation Handling:
Simulates employee cancellations and uses an NLP module for text classification to detect cancellations. Replacements are recommended using the trained policy network.

Requirements
Python 3.x

PyTorch

NumPy

Matplotlib

An NLP module providing the classify_text function

Installation
Clone the repository and install the required packages using your preferred package manager.

Usage
Run the main simulation script to start training and scheduling. You can adjust parameters such as the number of employees, shifts, training episodes, and evaluation intervals to suit your needs.

Project Structure
Data Models:
Contains definitions for the Employee and Shift classes.

Feature Encoders:
Functions to convert shift and employee attributes into numerical feature vectors.

Policy Network and Training:
Defines the neural network architecture and implements the training loop using the REINFORCE algorithm.

Simulation and Evaluation:
Functions to generate synthetic data, simulate shift scheduling, handle cancellations, recommend replacements, and visualize training metrics.

Visualization:
Generates plots for training rewards and evaluation metrics, which are saved for review.
