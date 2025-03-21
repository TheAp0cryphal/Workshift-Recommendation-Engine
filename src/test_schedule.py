import unittest
import pandas as pd
from scheduling import score_employee, match_employee

class TestSchedulingEngine(unittest.TestCase):
    def setUp(self):
        # Load employee data from CSV
        self.employees = pd.read_csv('data/employees.csv')

        # Define a synthetic shift where the required skills and day are provided.
        self.shift = {
            'required_skills': 'A,B',
            'day': 'Fri',
        }

    def test_score_employee(self):
        # Test scoring for a single employee (Employee_1)
        employee = self.employees.iloc[7] #Employee number 8
        score = score_employee(employee, self.shift['required_skills'], self.shift['day'])
        self.assertEqual(score, 5, "Employee_8's score should be 5. skill: 2, workload: 2, points:1")

    def test_match_employee(self):
        # Test overall matching: we expect the best candidate to be Employee_2.
        candidate = match_employee(self.employees, self.shift)
        self.assertEqual(candidate[0], 11, "The 11th Employee should be ideal candidate")

if __name__ == '__main__':
    unittest.main()