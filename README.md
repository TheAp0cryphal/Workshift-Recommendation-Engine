![ShiftScheduler drawio](https://github.com/user-attachments/assets/b717ff29-5ed1-4d94-a32a-fd06eba29023)

# Shift Scheduling Simulation: Project Documentation

# Overview

This document provides a comprehensive overview of a reinforcement
learning (RL) based shift scheduling simulation. The project focuses on
scheduling employee shifts by taking into account employee skills,
availability, reliability, and workload balance. In addition, it handles
shift cancellations and recommends replacements using an ML-based
ranking mechanism.

# Project Structure

The project is modularized into the following key components:

-   **Global Settings and Constants**: Definition of days, times, and
    required skills.

-   **Data Models**: Classes for `Employee` and `Shift`.

-   **Feature Encoding Functions**: Methods to convert shifts and
    employee attributes into numerical feature vectors.

-   **Policy Network**: A neural network (using PyTorch) that scores
    employee-shift pairings.

-   **RL Environment and Training**: Implements the simulation
    environment and uses policy gradients (REINFORCE) for training.

-   **Visualization and Metrics**: Functions for evaluation, plotting
    training progress, and computing key metrics.

-   **ML-Based Replacement Ranking**: Mechanism to recommend
    replacements for cancelled shifts.

-   **Synthetic Data Generation**: Functions to generate synthetic
    employees and shifts.

# Module Details

## Global Settings and Constants

**Constants:**

-   `DAYS`: A list of weekdays (e.g., Monday to Friday).

-   `TIMES`: Time slots (e.g., Morning, Evening).

-   `SKILLS`: Different skills required (e.g., Customer Service,
    Technical Support, Sales, Management).

## Data Models

**Employee Class:**

-   **Attributes:**

    -   `id` & `name`

    -   `skills` (subset of SKILLS)

    -   `availability` (subset of DAYS)

    -   `reliability` (probability of showing up)

    -   `points` (performance tracking)

    -   `assigned_shifts` (list of shifts assigned)

**Shift Class:**

-   **Attributes:**

    -   `id`, `day`, `time`

    -   `required_skill`

    -   `assigned_employee` (initially `None`)

## Feature Encoding Functions

The project provides functions to convert both shifts and employees into
numerical feature vectors:

-   `encode_shift(shift)`: One-hot encodes the day, time, and required
    skill.

-   `encode_employee(emp, current_workload)`: One-hot encodes employee
    skills and availability, and includes workload, reliability, and
    normalized points.

-   `get_feature_vector(shift, emp, emp_workload)`: Concatenates the
    shift and employee feature vectors.

## Policy Network

The `PolicyNetwork` is a simple feedforward neural network built using
PyTorch:

-   **Input Layer:** Receives the concatenated feature vector.

-   **Hidden Layer:** Applies a ReLU activation function.

-   **Output Layer:** Produces a scalar score that represents the
    suitability of an employee for a given shift.

## Reinforcement Learning Environment and Training

The `ShiftSchedulingEnv` class encapsulates the entire scheduling
environment:

-   **Episode Execution:** Randomizes the order of shifts and assigns
    employees based on the scores from the policy network.

-   **Reward Function:** Rewards valid assignments (skill match and
    availability) and penalizes invalid assignments and overtime.

-   **Policy Update:** Uses the REINFORCE algorithm to update the
    network weights based on the trajectory of (log probability, reward)
    pairs.

## Visualization and Metrics

Several functions are provided to evaluate and visualize the training
progress:

-   `evaluate_policy`: Measures metrics such as shift coverage, workload
    standard deviation, and skill match rate.

-   `plot_training_rewards` and `plot_evaluation_metrics`: Use
    Matplotlib to generate and save plots that display training
    progress.

## ML-Based Replacement Ranking

When an employee cancels a shift, the function `recommend_replacement`
uses the trained policy network to score potential replacement
candidates and selects the best one from those who have the required
skills and availability.

## Synthetic Data Generation

Synthetic data is generated using:

-   `generate_employees(n)`: Creates employees with random subsets of
    skills and availability.

-   `generate_shifts(n)`: Generates shifts with randomly assigned day,
    time, and required skill.

# Simulation Workflow

The simulation follows these steps:

1.  **Initialization:** Generate synthetic data (employees and shifts)
    and initialize the policy network along with the environment.

2.  **Training:** Run multiple episodes where the policy is trained via
    REINFORCE.

3.  **Evaluation:** After training, evaluate the policy by assigning
    shifts and measuring metrics.

4.  **Handling Cancellations:** Simulate cancellations using pre-defined
    messages and detect them with an NLP module. Penalize the employee
    and remove the assignment.

5.  **Replacement:** Recommend replacements for cancelled shifts using
    the ML-based ranking.

6.  **Reporting:** Compute final metrics such as final shift coverage,
    workload distribution, and output comprehensive simulation results.

# Usage Instructions

## Dependencies

The project requires the following:

-   Python 3.x

-   PyTorch

-   NumPy

-   Matplotlib

-   An NLP module (`nlp_module`) for classifying cancellation messages

## Running the Simulation

1.  Install all dependencies.

2.  Execute the main script:

    ``` {.bash language="bash"}
    python shift_scheduling_simulation.py
    ```
3.  The script will train the RL policy, generate plots in the `plots`
    directory, and print the simulation results.

# Conclusion

This project demonstrates a reinforcement learning approach to solving
the shift scheduling problem by incorporating employee characteristics,
skill requirements, and dynamic adjustments for cancellations. Its
modular design allows for easy maintenance and future improvements, such
as more sophisticated cancellation detection and adaptive scheduling
strategies.
