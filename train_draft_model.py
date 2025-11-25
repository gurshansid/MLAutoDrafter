"""
Training script for Fantasy Football Draft Neural Network
Uses actual 2024 stats and your DraftSimulator with position limits
FIXED: NumPy compatibility and tensor conversion issues
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

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class NeuralDraftAgent:
    """
    Agent that uses neural network to draft players
    Integrates with your DraftSimulator
    """
    
    def __init__(self, model, condensed_data_path='nfl_players_condensed.csv'):
        """
        Initialize agent
        
        Args:
            model: trained PlayerValueNetwork
            condensed_data_path: path to condensed player features
        """
        self.model = model
        
        # Load condensed data (has historical features)
        self.condensed_df = pd.read_csv(condensed_data_path)
        
        # Feature columns (exclude names)
        self.feature_columns = [col for col in self.condensed_df.columns 
                               if col not in ['first_name', 'last_name']]
        
        # CRITICAL FIX: Convert all feature columns to numeric types
        for col in self.feature_columns:
            self.condensed_df[col] = pd.to_numeric(self.condensed_df[col], errors='coerce')
        
        # Fill any NaN values with 0
        self.condensed_df[self.feature_columns] = self.condensed_df[self.feature_columns].fillna(0)
        
        print(f"Agent loaded with {len(self.condensed_df)} players")
        print(f"Features ({len(self.feature_columns)}): {self.feature_columns[:5]}...")
    
    def get_player_features(self, player_row, round_num):
        """
        Extract features for a player including round context
        
        Args:
            player_row: pandas Series from DraftSimulator's available_players
            round_num: current draft round
        
        Returns:
            tensor of features
        """
        # Match player from simulator to condensed data
        condensed_player = self.condensed_df[
            (self.condensed_df['first_name'] == player_row['first_name']) &
            (self.condensed_df['last_name'] == player_row['last_name'])
        ]
        
        if len(condensed_player) == 0:
            # Player not in condensed data (shouldn't happen but handle it)
            # Return zeros
            return torch.zeros(1 + len(self.feature_columns))
        
        # Get features and explicitly convert to float64
        features = condensed_player.iloc[0][self.feature_columns].values.astype(np.float64)
        
        # Verify no NaN values
        if np.any(np.isnan(features)):
            print(f"Warning: NaN values found for {player_row['first_name']} {player_row['last_name']}")
            features = np.nan_to_num(features, nan=0.0)
        
        # Add round number as float
        features_with_round = np.concatenate([[float(round_num)], features])
        
        # Now safely convert to tensor
        return torch.FloatTensor(features_with_round)
    
    def draft_player(self, simulator, team_id, round_num, training=True, epsilon=0.1):
        """
        Use neural network to select a player
        
        Args:
            simulator: DraftSimulator instance
            team_id: our team ID
            round_num: current round number
            training: if True, use epsilon-greedy exploration
            epsilon: exploration rate
        
        Returns:
            selected player (pandas Series) and log_prob (for training)
        """
        # Get available players that satisfy position constraints
        available = simulator.get_available_players()
        
        if len(available) == 0:
            return None, None
        
        # Filter for positions we can actually draft
        valid_players = []
        valid_indices = []
        
        for idx, player in available.iterrows():
            if simulator.can_draft_position(team_id, player['position']):
                valid_players.append(player)
                valid_indices.append(idx)
        
        if len(valid_players) == 0:
            print(f"  WARNING: Team {team_id} has no valid positions!")
            return None, None
        
        # Evaluate each valid player with neural network
        scores = []
        for player in valid_players:
            features = self.get_player_features(player, round_num)
            score = self.model(features.unsqueeze(0))
            scores.append(score)
        
        # Stack scores
        scores_tensor = torch.cat(scores).squeeze()
        
        # Convert to probabilities
        probs = torch.softmax(scores_tensor, dim=0)
        
        # Choose player (epsilon-greedy for training)
        if training and random.random() < epsilon:
            # Explore: random pick
            pick_idx = random.randint(0, len(valid_players) - 1)
        else:
            # Exploit
            if training:
                pick_idx = torch.multinomial(probs, 1).item()
            else:
                pick_idx = torch.argmax(probs).item()
        
        # Get log probability for training
        log_prob = None
        if training:
            log_prob = torch.log(probs[pick_idx] + 1e-10)
        
        # Return the selected player
        selected_player = valid_players[pick_idx]
        
        return selected_player, log_prob


def train_model(model, 
                num_episodes=1500,
                learning_rate=0.001,
                n_teams=12,
                n_rounds=9):
    """
    Train the model using your DraftSimulator and actual 2024 stats
    
    Args:
        model: PlayerValueNetwork
        num_episodes: number of training episodes
        learning_rate: learning rate
        n_teams: number of teams in draft
        n_rounds: number of rounds
    
    Returns:
        tuple: (episode_scores, episode_positions, episode_ranks)
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize agent
    agent = NeuralDraftAgent(model)
    
    # Storage for results
    episode_scores = []
    episode_positions = []
    episode_ranks = []
    best_score = 0
    
    print(f"\n{'='*60}")
    print("Starting Training with Real 2024 Data")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Draft: {n_teams} teams, {n_rounds} rounds")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        # Random draft position each episode
        our_position = random.randint(0, n_teams - 1)
        
        # Decay exploration (slower decay for more exploration)
        epsilon = max(0.05, 0.4 - (episode / num_episodes) * 0.30)
        
        # Create simulator
        simulator = DraftSimulator(
            player_data_path='nfl_player_data_with_history.csv',
            n_teams=n_teams,
            n_rounds=n_rounds
        )
        
        # Track log probabilities for our picks
        log_probs = []
        
        # Simulate draft
        model.train()
        
        for round_num in range(1, n_rounds + 1):
            simulator.current_round = round_num
            draft_order = simulator.get_draft_order(round_num)
            
            for team_id in draft_order:
                if team_id == our_position:
                    # OUR TURN - use neural network
                    player, log_prob = agent.draft_player(
                        simulator, 
                        team_id, 
                        round_num,
                        training=True,
                        epsilon=epsilon
                    )
                    
                    if player is not None:
                        simulator.make_pick(team_id, player)
                        if log_prob is not None:
                            log_probs.append(log_prob)
                
                else:
                    # OTHER TEAM - use ADP-based drafting
                    player = simulator.draft_player_by_adp(team_id)
                    if player is not None:
                        simulator.make_pick(team_id, player)
        
        # Evaluate all teams using ACTUAL 2024 performance
        scores = simulator.evaluate_draft()
        our_score = scores[our_position]
        
        # Get our rank (1 = best, 12 = worst)
        sorted_scores = sorted(scores.values(), reverse=True)
        our_rank = sorted_scores.index(our_score) + 1
        
        # Record results
        episode_scores.append(our_score)
        episode_positions.append(our_position + 1)  # 1-indexed for display
        episode_ranks.append(our_rank)
        
        # Update best score
        if our_score > best_score:
            best_score = our_score
            torch.save(model.state_dict(), 'best_draft_model.pth')
        
        # Calculate reward (stronger signal)
        baseline = np.mean(list(scores.values()))
        
        # Normalized score difference (stronger weight)
        score_reward = (our_score - baseline) / 100
        
        # Rank reward (exponential to heavily reward top finishes)
        # 1st = 1.0, 6th = 0.0, 12th = -1.0
        rank_reward = (n_teams - our_rank) / (n_teams / 2) - 1.0
        
        # Combined reward (70% rank, 30% score)
        reward = 0.3 * score_reward + 0.7 * rank_reward
        
        # Bonus for top 3 finish
        if our_rank <= 3:
            reward += 0.5
        
        # Policy gradient loss
        if len(log_probs) > 0:
            loss = 0
            for log_prob in log_probs:
                loss += -log_prob * reward
            loss = loss / len(log_probs)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            loss = torch.tensor(0.0)
        
        # Logging
        if (episode + 1) % 25 == 0:
            avg_score = np.mean(episode_scores[-25:])
            avg_rank = np.mean(episode_ranks[-25:])
            print(f"Ep {episode+1:4d}/{num_episodes} | "
                  f"Pos:{our_position+1:2d} | "
                  f"Score:{our_score:5.1f} | "
                  f"Rank:{our_rank:2d}/{n_teams} | "
                  f"Avg(25):{avg_score:5.1f} | "
                  f"AvgRank:{avg_rank:.1f} | "
                  f"Best:{best_score:5.1f} | "
                  f"ε:{epsilon:.3f}")
    
    return episode_scores, episode_positions, episode_ranks


def plot_training_results(episode_scores, episode_positions, episode_ranks, 
                          n_teams=12, save_path='training_results.png'):
    """
    Plot training results
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Score over time
    ax = axes[0, 0]
    ax.plot(episode_scores, alpha=0.3, label='Episode Score')
    window = 25
    moving_avg = pd.Series(episode_scores).rolling(window=window).mean()
    ax.plot(moving_avg, label=f'{window}-Episode MA', linewidth=2, color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Fantasy Points (2024 Actual)')
    ax.set_title('Training Progress: Team Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Rank over time
    ax = axes[0, 1]
    ax.plot(episode_ranks, alpha=0.3, label='Episode Rank')
    moving_avg_rank = pd.Series(episode_ranks).rolling(window=window).mean()
    ax.plot(moving_avg_rank, label=f'{window}-Episode MA', linewidth=2, color='red')
    ax.axhline(y=n_teams/2, color='gray', linestyle='--', label='Median')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Finish Rank (1=Best)')
    ax.set_title('Training Progress: Draft Rank')
    ax.set_ylim(n_teams + 0.5, 0.5)  # Invert y-axis (1 at top)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Score distribution
    ax = axes[1, 0]
    ax.hist(episode_scores, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(episode_scores), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(episode_scores):.1f}')
    ax.set_xlabel('Fantasy Points')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Rank distribution
    ax = axes[1, 1]
    rank_counts = [episode_ranks.count(r) for r in range(1, n_teams + 1)]
    ax.bar(range(1, n_teams + 1), rank_counts, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Finish Rank')
    ax.set_ylabel('Frequency')
    ax.set_title('Rank Distribution')
    ax.set_xticks(range(1, n_teams + 1))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Training results plot saved to '{save_path}'")


def main():
    """
    Main training function
    """
    print("="*60)
    print("Fantasy Football Draft - Training with Real 2024 Stats")
    print("="*60)
    
    # Load condensed data to get input size
    condensed_df = pd.read_csv('nfl_players_condensed.csv')
    feature_columns = [col for col in condensed_df.columns 
                      if col not in ['first_name', 'last_name']]
    
    input_size = 1 + len(feature_columns)  # round + features
    
    print(f"\nInput size: {input_size} features")
    print(f"  - 1 round context")
    print(f"  - {len(feature_columns)} player features")
    
    # Create model
    model = create_player_value_network(input_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters (~{total_params*4/1024:.1f} KB)")
    
    # Train
    num_episodes = 1500
    episode_scores, episode_positions, episode_ranks = train_model(
        model=model,
        num_episodes=num_episodes,
        learning_rate=0.001,
        n_teams=12,
        n_rounds=9
    )
    
    # Save final model
    torch.save(model.state_dict(), 'final_draft_model.pth')
    print(f"\n✅ Final model saved to 'final_draft_model.pth'")
    
    # Plot results
    plot_training_results(episode_scores, episode_positions, episode_ranks)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Total episodes: {num_episodes}")
    print(f"Best score: {max(episode_scores):.1f} points")
    print(f"Mean score: {np.mean(episode_scores):.1f} points")
    print(f"Std dev: {np.std(episode_scores):.1f}")
    print(f"Final 50 avg: {np.mean(episode_scores[-50:]):.1f} points")
    
    # Rank analysis
    print(f"\n{'='*60}")
    print("Rank Analysis")
    print(f"{'='*60}")
    print(f"Mean rank: {np.mean(episode_ranks):.2f} / 12")
    print(f"Best rank: {min(episode_ranks)}")
    print(f"Worst rank: {max(episode_ranks)}")
    print(f"Final 50 avg rank: {np.mean(episode_ranks[-50:]):.2f}")
    
    # Top 3 finishes
    top3_finishes = sum(1 for r in episode_ranks if r <= 3)
    print(f"\nTop 3 finishes: {top3_finishes} / {num_episodes} ({100*top3_finishes/num_episodes:.1f}%)")
    
    # Win rate
    wins = sum(1 for r in episode_ranks if r == 1)
    print(f"Wins (1st place): {wins} / {num_episodes} ({100*wins/num_episodes:.1f}%)")
    
    print(f"\n{'='*60}")
    print("✅ Training Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()