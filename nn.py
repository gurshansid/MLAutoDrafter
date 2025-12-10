# File contains both NN, player and position

import torch
import torch.nn as nn
import numpy as np
import random

from envirement import DraftEnvironment
from features import get_qb_features, get_rb_features, get_wr_features, get_te_features



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
    
class HierarchicalAgent:

    def __init__(self, position_model, player_models, env: DraftEnvironment):
        self.position_model = position_model
        self.player_models = player_models
        self.env = env
        self.device = next(position_model.parameters()).device

        self.positions = ["QB", "RB", "WR", "TE"]
        self.pos_to_idx = {p: i for i, p in enumerate(self.positions)}

        print("[Agent] Hierarchical agent initialized")

    def pick_player(
        self,
        env: DraftEnvironment,
        team_id: int,
        round_num: int,
        epsilon: float,
        training: bool = True,
    ):

        if len(env.rosters[team_id]) >= env.max_players_per_team:
            return None, None

        context = env.get_context_vector(team_id, round_num)
        context_tensor = torch.tensor(
            context, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        legal_positions = []
        legal_indices = []

        for pos in self.positions:
            if env.get_available_at_position(team_id, pos):
                legal_positions.append(pos)
                legal_indices.append(self.pos_to_idx[pos])

        if not legal_positions:
            return None, None

        self.position_model.train() if training else self.position_model.eval()

        with torch.set_grad_enabled(training):
            position_logits = self.position_model(context_tensor).squeeze(0)

            legal_mask = torch.full((4,), float("-inf"), device=self.device)
            for idx in legal_indices:
                legal_mask[idx] = 0.0

            masked_logits = position_logits + legal_mask
            position_probs = torch.softmax(masked_logits, dim=0)

            if training and random.random() < epsilon:
                chosen_pos_idx = random.choice(legal_indices)
            else:
                chosen_pos_idx = int(torch.argmax(position_probs).item())

            position_log_prob = (
                torch.log(position_probs[chosen_pos_idx] + 1e-10)
                if training
                else None
            )

        chosen_position = self.positions[chosen_pos_idx]

        available_players = env.get_available_at_position(team_id, chosen_position)
        if not available_players:
            return None, None

        player_features = []
        for player_row in available_players:
            if chosen_position == "QB":
                feats = get_qb_features(
                    player_row, env.feature_means, env.feature_stds
                )
            elif chosen_position == "RB":
                feats = get_rb_features(
                    player_row, env.feature_means, env.feature_stds
                )
            elif chosen_position == "WR":
                feats = get_wr_features(
                    player_row, env.feature_means, env.feature_stds
                )
            else:  # TE
                feats = get_te_features(
                    player_row, env.feature_means, env.feature_stds
                )
            player_features.append(feats)

        player_tensor = torch.tensor(
            np.stack(player_features), dtype=torch.float32, device=self.device
        )
        context_batch = context_tensor.repeat(len(available_players), 1)

        player_model = self.player_models[chosen_position]
        player_model.train() if training else player_model.eval()

        with torch.set_grad_enabled(training):
            player_values = player_model(player_tensor, context_batch).squeeze(1)
            player_probs = torch.softmax(player_values, dim=0)

            if training and random.random() < epsilon:
                player_idx = random.randrange(len(available_players))
            else:
                player_idx = int(torch.argmax(player_probs).item())

            player_log_prob = (
                torch.log(player_probs[player_idx] + 1e-10) if training else None
            )

        chosen_player = available_players[player_idx]

        total_log_prob = None
        if (
            training
            and position_log_prob is not None
            and player_log_prob is not None
        ):
            total_log_prob = position_log_prob + player_log_prob

        return chosen_player, total_log_prob