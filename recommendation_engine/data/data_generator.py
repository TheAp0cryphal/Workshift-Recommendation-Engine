"""
Functions for generating synthetic data (employees and shifts).
"""
import random
import numpy as np
from recommendation_engine.core.constants import DAYS, TIMES, SKILLS
from recommendation_engine.data.models import Employee, Shift

def generate_employees(n=10):
    """
    Generate n synthetic employees with random attributes.
    """
    employees = []
    names = [f"Employee_{i}" for i in range(n)]
    
    for i in range(n):
        # Random skills (1-4 skills per employee)
        num_skills = random.randint(1, len(SKILLS))
        skills = random.sample(SKILLS, num_skills)
        
        # Random availability (2-5 days per employee)
        num_available_days = random.randint(2, len(DAYS))
        availability = random.sample(DAYS, num_available_days)
        
        # Random reliability (0.7-1.0)
        reliability = 0.7 + random.random() * 0.3
        
        emp = Employee(i, names[i], skills, availability, reliability)
        employees.append(emp)
    
    return employees

def generate_shifts(n=20):
    """
    Generate n synthetic shifts with random attributes.
    """
    shifts = []
    
    for i in range(n):
        day = random.choice(DAYS)
        time = random.choice(TIMES)
        required_skill = random.choice(SKILLS)
        
        shift = Shift(i, day, time, required_skill)
        shifts.append(shift)
    
    return shifts 