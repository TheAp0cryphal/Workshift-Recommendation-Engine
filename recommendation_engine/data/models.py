"""
Data models for the recommendation engine.
"""
from recommendation_engine.core.constants import DAYS

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