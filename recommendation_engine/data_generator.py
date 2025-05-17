"""
Functions for generating synthetic data for the recommendation engine.
"""
import random
from recommendation_engine.constants import DAYS, TIMES, SKILLS
from recommendation_engine.models import Employee, Shift

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