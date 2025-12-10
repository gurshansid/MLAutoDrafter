import os
import torch
import numpy as np
import random


from training import train_on_2023
from evalution import plot_training, evaluate_on_2024, inspect_learned_policy

from envirement import DraftEnvironment

def main():
    if not os.path.exists("nfl_players_condensed.csv"):
        print("ERROR: Missing nfl_players_condensed.csv "
              "(build it with build_condensed_from_2023.py)")
        return

    if not os.path.exists("nfl_players_2023_stats.csv"):
        print("ERROR: Missing nfl_players_2023_stats.csv "
              "(build it with your collect_data/player_condenser pipeline)")
        return

    if not os.path.exists("nfl_players_2024_stats.csv"):
        print("ERROR: Missing nfl_players_2024_stats.csv "
              "(build it with get_2024.py or similar)")
        return

    random_seed = np.random.randint(0, 1_000_000)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    print("\n" + "=" * 80)
    print("HIERARCHICAL RL – TRAIN 2023, TEST 2024")
    print("=" * 80)
    print(f"Random seed: {random_seed}")
    print("=" * 80)

    # Train on 2023
    position_model, player_models, env_2023, scores, ranks = train_on_2023(
        total_episodes=1500,
        learning_rate=1e-3,
        condensed_path="nfl_players_condensed.csv",
        stats_2023_path="nfl_players_2023_stats.csv",
    )

    plot_training(scores, ranks)

    scores_2024, ranks_2024 = evaluate_on_2024(
        position_model,
        player_models,
        condensed_path="nfl_players_condensed.csv",
        stats_2024_path="nfl_players_2024_stats.csv",
        n_eval_drafts=500,
    )

    env_2024 = DraftEnvironment(
        condensed_path="nfl_players_condensed.csv",
        stats_path="nfl_players_2024_stats.csv",
        n_teams=12,
        n_rounds=8,
        seed=999,
    )
    print("\n[INSPECT] Learned player preferences on 2024 env:")
    inspect_learned_policy(position_model, player_models, env_2024)


if __name__ == "__main__":
    main()