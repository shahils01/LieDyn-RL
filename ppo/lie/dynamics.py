import torch
import torch.nn as nn


class LieDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, algebra_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, algebra_dim),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))
