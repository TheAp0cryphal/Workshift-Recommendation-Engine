"""
Neural network model for the policy network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from recommendation_engine.core.constants import DAYS, TIMES, SKILLS

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=(len(DAYS)+len(TIMES)+len(SKILLS)) + (len(SKILLS)+len(DAYS)+3), hidden_dim=32):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Outputs a scalar score

    def forward(self, x):
        x = F.relu(self.fc1(x))
        score = self.fc2(x)
        return score 