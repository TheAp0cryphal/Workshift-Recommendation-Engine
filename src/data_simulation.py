import pandas as pd
import numpy as np

# Define a mapping for skills to letters
skill_mapping = {
    'Cashiering': 'A',
    'Customer Service': 'B',
    'Inventory Management': 'C',
    'Food Preparation': 'D'
}

def generate_employees(num=10):
    # Use the skill mapping to generate skills
    possible_skills = list(skill_mapping.keys())
    
    employees = pd.DataFrame({
        'id': range(1, num+1),
        'name': [f'Employee_{i}' for i in range(1, num+1)],
        # Randomly assign 2 skills per employee from the list, then map to letters
        'skills': [', '.join([skill_mapping[s] for s in np.random.choice(possible_skills, 2, replace=False)]) for _ in range(num)],
        # 1 day availability for prototype level complexity
        'availability': [np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri']) for _ in range(num)],
        #Randomly assigned workload, ideally would be derived from a shift table that counts number of occurences
        'current_workload': np.random.randint(0, 5, size=num),
        'points': 100  # starting base points
    })
    employees.to_csv('data/employees.csv', index=False)
    return employees

def generate_shifts(num=5):
    # Use the skill mapping to generate skills
    possible_skills = list(skill_mapping.keys())
    
    shifts = pd.DataFrame({
        'shift_id': range(1, num+1),
        # Randomly assign 2 skills per shift from the list, then map to letters
        'required_skills': [', '.join([skill_mapping[s] for s in np.random.choice(possible_skills, 2, replace=False)]) for _ in range(num)],
        'day': [np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri']) for _ in range(num)],
        'time_slot': ['09:00-17:00' for _ in range(num)]
    })
    shifts.to_csv('data/shifts.csv', index=False)
    return shifts

if __name__ == "__main__":
    print("Generating synthetic data...")
    employees = generate_employees()
    shifts = generate_shifts()
    print("Data generated and saved in the data/ directory.")
