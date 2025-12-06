# File contains both NN, player and position

import torch
import torch.nn as nn


# Position NN

class PositionPolicyNetwork(nn.Module):
    
    def __init__(self, context_size: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)

# Player NN

class PlayerValueNetwork(nn.Module):

    def __init__(self, player_feature_size: int, context_size: int = 10):
        super().__init__()

        self.player_encoder = nn.Sequential(
            nn.Linear(player_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.decision_head = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, player_features: torch.Tensor, context: torch.Tensor):
        player_enc = self.player_encoder(player_features)
        context_enc = self.context_encoder(context)
        combined = torch.cat([player_enc, context_enc], dim=1)
        return self.decision_head(combined)