"""
Neural Network for Fantasy Football Player Valuation
Simple player evaluator - takes player features, outputs value score
"""

import torch
import torch.nn as nn


class PlayerValueNetwork(nn.Module):
    """
    Neural network that evaluates a single player's value
    
    Given player features (stats, age, position, team, round context),
    outputs a single score representing how valuable this player is
    """
    
    def __init__(self, input_size, hidden1_size=128, hidden2_size=64):
        """
        Initialize the network
        
        Args:
            input_size: number of input features
            hidden1_size: neurons in first hidden layer (default 128)
            hidden2_size: neurons in second hidden layer (default 64)
        """
        super(PlayerValueNetwork, self).__init__()
        
        # Input → Hidden Layer 1
        self.fc1 = nn.Linear(input_size, hidden1_size)
        
        # Hidden Layer 1 → Hidden Layer 2
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        
        # Hidden Layer 2 → Output (single value score)
        self.fc3 = nn.Linear(hidden2_size, 1)
        
        # Activation function (ReLU)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass: player features → value score
        
        Args:
            x: tensor of player features [batch_size, input_size]
        
        Returns:
            value score [batch_size, 1]
        """
        # Pass through first layer with ReLU
        x = self.relu(self.fc1(x))
        
        # Pass through second layer with ReLU
        x = self.relu(self.fc2(x))
        
        # Output layer (no activation - raw score)
        x = self.fc3(x)
        
        return x


def create_player_value_network(input_size, hidden1=128, hidden2=64):
    """
    Factory function to create a player value network
    
    Args:
        input_size: number of input features
        hidden1: size of first hidden layer
        hidden2: size of second hidden layer
    
    Returns:
        PlayerValueNetwork instance
    """
    return PlayerValueNetwork(input_size, hidden1, hidden2)


def count_parameters(model):
    """
    Count total trainable parameters in the model
    
    Args:
        model: PyTorch model
    
    Returns:
        number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("Player Value Network - Architecture Test")
    print("=" * 60)
    
    # Example: input features
    # [round_num, pos_qb, pos_rb, pos_wr, pos_te, team_win_pct, 
    #  adp, age, is_rookie, draft_round, seasons, games, completions,
    #  attempts, passing_yards, passing_tds, interceptions, carries,
    #  rushing_yards, rushing_tds, receptions, targets, receiving_yards,
    #  receiving_tds, points_per_game]
    
    input_size = 25  # 1 (round) + 24 (player features)
    
    # Create model
    model = create_player_value_network(input_size)
    
    print(f"\nInput size: {input_size} features")
    print(f"\nModel Architecture:")
    print(model)
    
    print(f"\nTotal parameters: {count_parameters(model):,}")
    print(f"Model size: ~{count_parameters(model) * 4 / 1024:.1f} KB")
    
    # Test with sample data
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    # Single player
    single_player = torch.randn(1, input_size)
    output = model(single_player)
    print(f"\nSingle player input shape: {single_player.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Value score: {output.item():.4f}")
    
    # Batch of 20 players (typical draft choice)
    batch_players = torch.randn(20, input_size)
    batch_output = model(batch_players)
    print(f"\nBatch of 20 players input shape: {batch_players.shape}")
    print(f"Output shape: {batch_output.shape}")
    print(f"\nValue scores:")
    for i, score in enumerate(batch_output[:5]):  # Show first 5
        print(f"  Player {i+1}: {score.item():.4f}")
    print("  ...")
    
    # Show how to pick best player
    best_idx = torch.argmax(batch_output)
    best_score = batch_output[best_idx].item()
    print(f"\nBest player: #{best_idx.item() + 1} with score {best_score:.4f}")
    
    # Show softmax probabilities
    probs = torch.softmax(batch_output.squeeze(), dim=0)
    print(f"\nTop 5 players by probability:")
    top5_probs, top5_idx = torch.topk(probs, 5)
    for i, (idx, prob) in enumerate(zip(top5_idx, top5_probs)):
        print(f"  {i+1}. Player #{idx.item() + 1}: {prob.item()*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ Network test complete!")
    print("=" * 60)