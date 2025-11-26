"""
Training script for Fantasy Football Draft Neural Network
Uses actual 2024 stats and your DraftSimulator with position limits
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import sys
from draft_network import create_player_value_network
import matplotlib.pyplot as plt

# Import your DraftSimulator
sys.path.append('.')
from draft_simulator import DraftSimulator

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class NeuralDraftAgent:
    """
    Original efficient agent that matches players by (first, last)
    """

    def __init__(self, model, condensed_data_path='nfl_players_condensed.csv'):
        self.model = model

        df = pd.read_csv(condensed_data_path)
        self.feature_columns = [
            c for c in df.columns if c not in ['first_name', 'last_name']
        ]

        # Normalize and numeric
        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[self.feature_columns] = df[self.feature_columns].fillna(0)

        print(f"Agent loaded with {len(df)} players")
        print(f"Features ({len(self.feature_columns)}): {self.feature_columns[:5]}...")

        # fast lookup: (first,last) → vector
        self.lookup = {}
        for _, row in df.iterrows():
            key = (row['first_name'], row['last_name'])
            self.lookup[key] = row[self.feature_columns].values.astype(np.float32)

    def get_player_features(self, player_row, round_num):
        key = (player_row['first_name'], player_row['last_name'])
        stats = self.lookup.get(key)

        if stats is None:
            return torch.zeros(1 + len(self.feature_columns))

        x = np.concatenate([[float(round_num)], stats])
        return torch.tensor(x, dtype=torch.float32)

    def draft_player(self, simulator, team_id, round_num, training=True, epsilon=0.1):
        available = simulator.get_available_players()
        if len(available) == 0:
            return None, None

        # Filter legal picks
        legal = []
        for _, p in available.iterrows():
            if simulator.can_draft_position(team_id, p['position']):
                legal.append(p)

        if len(legal) == 0:
            return None, None

        # Score players
        scores = []
        for p in legal:
            f = self.get_player_features(p, round_num)
            scores.append(self.model(f.unsqueeze(0)))

        logits = torch.cat(scores).squeeze()
        probs = torch.softmax(logits, dim=0)

        # ε-greedy
        if training and random.random() < epsilon:
            idx = random.randrange(len(legal))
        else:
            idx = probs.argmax().item()

        log_prob = torch.log(probs[idx] + 1e-10) if training else None

        return legal[idx], log_prob


def train_model(model, num_episodes=100, learning_rate=0.001,
                n_teams=12, n_rounds=9):

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        agent = NeuralDraftAgent(model)

        episode_scores = []
        episode_positions = []
        episode_ranks = []
        best_score = 0

        print("\n" + "=" * 60)
        print("Starting Training with Real 2024 Data")
        print("=" * 60)
        print(f"Episodes: {num_episodes}")
        print(f"Learning rate: {learning_rate}")
        print(f"Draft: {n_teams} teams, {n_rounds} rounds")
        print("=" * 60 + "\n")

        for ep in range(num_episodes):
            our_position = random.randint(0, n_teams - 1)
            epsilon = max(0.05, 0.4 - (ep / num_episodes) * 0.30)

            simulator = DraftSimulator(
                player_data_path='nfl_player_data_with_history.csv',
                n_teams=n_teams,
                n_rounds=n_rounds
            )

            log_probs = []
            model.train()

            # FULL draft
            for rnd in range(1, n_rounds + 1):
                simulator.current_round = rnd
                order = simulator.get_draft_order(rnd)

                for team_id in order:
                    if team_id == our_position:
                        p, lp = agent.draft_player(
                            simulator,
                            team_id,
                            rnd,
                            training=True,
                            epsilon=epsilon
                        )
                        if p is not None:
                            simulator.make_pick(team_id, p)
                        if lp is not None:
                            log_probs.append(lp)
                    else:
                        p = simulator.draft_player_by_adp(team_id)
                        if p is not None:
                            simulator.make_pick(team_id, p)

            # SCORE VIA evaluate_draft (original sim)
            scores = simulator.evaluate_draft()
            our_score = scores[our_position]

            sorted_scores = sorted(scores.values(), reverse=True)
            our_rank = sorted_scores.index(our_score) + 1

            episode_scores.append(our_score)
            episode_positions.append(our_position + 1)
            episode_ranks.append(our_rank)

            # reward shaping
            baseline = np.mean(list(scores.values()))
            score_reward = (our_score - baseline) / 100
            rank_reward = (n_teams - our_rank) / (n_teams / 2) - 1
            reward = 0.3 * score_reward + 0.7 * rank_reward

            if our_rank <= 3:
                reward += 0.5

            if log_probs:
                loss = -(torch.stack(log_probs) * reward).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            best_score = max(best_score, our_score)

            # --- PRINT EVERY EP ---
            avg_last = (
                np.mean(episode_scores[-25:])
                if len(episode_scores) >= 25
                else np.mean(episode_scores)
            )
            avg_rank = np.mean(episode_ranks)

            print(
                f"Ep {ep+1:4d}/{num_episodes} | "
                f"Pos:{our_position+1:2d} | "
                f"Score:{our_score:6.1f} | "
                f"Rank:{our_rank:2d}/{n_teams} | "
                f"Avg({min(len(episode_scores),25)}):{avg_last:6.1f} | "
                f"AvgRank:{avg_rank:4.1f} | "
                f"Best:{best_score:6.1f} | "
                f"ε:{epsilon:.3f}"
            )

        return episode_scores, episode_positions, episode_ranks


def main():
    df = pd.read_csv('nfl_players_condensed.csv')
    feature_columns = [
        c for c in df.columns if c not in ['first_name', 'last_name']
    ]
    input_size = 1 + len(feature_columns)

    print("=" * 60)
    print("Fantasy Football Draft - Training with Real 2024 Stats")
    print("=" * 60)
    print(f"Input size: {input_size}")
    print(f"Model features: {len(feature_columns)}")
    print("=" * 60)

    model = create_player_value_network(input_size)

    scores, positions, ranks = train_model(
        model=model,
        num_episodes=100,
        learning_rate=0.001,
        n_teams=12,
        n_rounds=9
    )

    torch.save(model.state_dict(), 'final_draft_model.pth')
    print("\nModel saved.")


if __name__ == "__main__":
    main()
