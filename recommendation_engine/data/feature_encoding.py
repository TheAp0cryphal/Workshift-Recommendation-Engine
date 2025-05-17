"""
Functions for encoding shifts and employees into feature vectors.
"""
import numpy as np
from recommendation_engine.core.constants import DAYS, TIMES, SKILLS

def encode_shift(shift):
    """
    Encode shift into a feature vector.
    Features:
      - Day one-hot (len = len(DAYS))
      - Time one-hot (len = len(TIMES))
      - Required skill one-hot (len = len(SKILLS))
    """
    day_vec = [1 if d == shift.day else 0 for d in DAYS]
    time_vec = [1 if t == shift.time else 0 for t in TIMES]
    skill_vec = [1 if s == shift.required_skill else 0 for s in SKILLS]
    return np.array(day_vec + time_vec + skill_vec, dtype=np.float32)

def encode_employee(emp, current_workload):
    """
    Encode employee into a feature vector.
    Features:
      - Skills one-hot (len = len(SKILLS))
      - Availability one-hot for each day (len = len(DAYS))
      - Current workload (normalized scalar)
      - Reliability (scalar)
      - Points (scalar, normalized by 100)
    """
    skill_vec = [1 if s in emp.skills else 0 for s in SKILLS]
    avail_vec = [1 if d in emp.availability else 0 for d in DAYS]
    workload = np.array([current_workload], dtype=np.float32)
    reliability = np.array([emp.reliability], dtype=np.float32)
    points = np.array([emp.points / 100.0], dtype=np.float32)
    return np.concatenate([np.array(skill_vec, dtype=np.float32),
                           np.array(avail_vec, dtype=np.float32),
                           workload, reliability, points])

def get_feature_vector(shift, emp, emp_workload):
    """
    Concatenate shift and employee features.
    """
    shift_feat = encode_shift(shift)
    emp_feat = encode_employee(emp, emp_workload)
    return np.concatenate([shift_feat, emp_feat]) 