# We score the employee based on direct matches, where 
# Skills match = +1 score
# Availability = Mon, Tue ... Fri
# Workload = Current Occupancy (1 - 5): Do I have to determine occupancy score?

#workload is arbitrary? based on what? We can add later...

def score_employee(employee, required_skills, shift_day):
    # Convert skills into a set for comparison
   
    emp_skills = set(employee['skills'].split(','))
    req_skills = set(required_skills.split(','))

    if employee['id'] == 8:
        print(emp_skills)
        print(req_skills)

    # Calculate skill match score
    skill_score = len(emp_skills.intersection(req_skills))
    
    # Check availability (binary score)
    availability_score = 1 if employee['availability'] == shift_day else 0
    
    # Lower workload and higher points are better
    workload_score = max(0, 5 - employee['current_workload'])
    points_score = employee['points'] / 100.0  # normalize score ranged: [0 - 1]

    # Weighted sum (weights can be adjusted)
    if employee['id'] == 8:
        print(availability_score, skill_score, workload_score, points_score)
    total_score = availability_score * ((2 * skill_score) + (1 * workload_score) + (1 * points_score))
    return total_score

def match_employee(employees, shift):
    scores = []
    for _, employee in employees.iterrows():
        score = score_employee(employee, shift['required_skills'], shift['day'])
        scores.append((employee['id'], score))
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0]  # Return the best candidate (employee id and score), heap can be used to boost fetch performance