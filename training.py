import torch
import torch.optim as optim
import numpy as np
import random
import copy

from envirement import DraftEnvironment, v as print_league_roster_counts
from nn import PositionPolicyNetwork, PlayerValueNetwork, HierarchicalAgent
from features import get_qb_features, get_rb_features, get_wr_features, get_te_features



def train_on_2023(
    total_episodes: int = 1500,
    learning_rate: float = 1e-3,
    condensed_path: str = "nfl_players_condensed.csv",
    stats_2023_path: str = "nfl_players_2023_stats.csv",
):

    print("=" * 80)
    print("HIERARCHICAL RL – TRAIN ON 2023, EVAL ON 2024")
    print("=" * 80)
    print("  • Features from nfl_players_condensed.csv (built from 2023)")
    print("  • Reward during training = 2023 fantasy_points_ppr")
    print("  • We will later freeze the policy and test on 2024.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_scores = []
    all_ranks = []

    best_avg_score = -float("inf")
    best_models = None


    env_train = DraftEnvironment(
        condensed_path=condensed_path,
        stats_path=stats_2023_path,
        n_teams=12,
        n_rounds=8,
        seed=42,
    )

    position_model = PositionPolicyNetwork(context_size=10).to(device)

    sample_qb = env_train.player_pool[env_train.player_pool["position"] == "QB"].iloc[0]
    sample_rb = env_train.player_pool[env_train.player_pool["position"] == "RB"].iloc[0]
    sample_wr = env_train.player_pool[env_train.player_pool["position"] == "WR"].iloc[0]
    sample_te = env_train.player_pool[env_train.player_pool["position"] == "TE"].iloc[0]

    qb_size = len(get_qb_features(sample_qb, env_train.feature_means, env_train.feature_stds))
    rb_size = len(get_rb_features(sample_rb, env_train.feature_means, env_train.feature_stds))
    wr_size = len(get_wr_features(sample_wr, env_train.feature_means, env_train.feature_stds))
    te_size = len(get_te_features(sample_te, env_train.feature_means, env_train.feature_stds))

    print(f"Feature sizes: QB={qb_size}, RB={rb_size}, WR={wr_size}, TE={te_size}")

    player_models = {
        "QB": PlayerValueNetwork(qb_size, context_size=10).to(device),
        "RB": PlayerValueNetwork(rb_size, context_size=10).to(device),
        "WR": PlayerValueNetwork(wr_size, context_size=10).to(device),
        "TE": PlayerValueNetwork(te_size, context_size=10).to(device),
    }

    position_optimizer = optim.Adam(position_model.parameters(), lr=learning_rate)
    player_optimizers = {
        pos: optim.Adam(model.parameters(), lr=learning_rate)
        for pos, model in player_models.items()
    }

    agent = HierarchicalAgent(position_model, player_models, env_train)

    print("\n[TRAIN] Starting training on 2023...\n")

    for episodes_done in range(total_episodes):

        epsilon = max(0.05, 0.5 - 0.45 * min(1.0, episodes_done / 1000))

        our_team = env_train.reset()
        log_probs = []


        for rnd in range(1, env_train.n_rounds + 1):
            env_train.current_round = rnd
            order = env_train.get_draft_order(rnd)

            for tid in order:
                if tid == our_team:
                    player, log_prob = agent.pick_player(
                        env_train, tid, rnd, epsilon, training=True
                    )
                    if player is not None:
                        env_train.make_pick(tid, player)
                        if log_prob is not None:
                            log_probs.append(log_prob)
                else:
                    player = env_train.bot_pick(tid)
                    if player is not None:
                        env_train.make_pick(tid, player)


        scores_dict = env_train.evaluate_league()
        our_score = scores_dict[our_team]
        league_scores = list(scores_dict.values())
        our_rank = sorted(league_scores, reverse=True).index(our_score) + 1

        league_mean = np.mean(league_scores)
        league_std = np.std(league_scores) if np.std(league_scores) > 0 else 1.0
        reward = (our_score - league_mean) / league_std


        if log_probs:
            loss = -(torch.stack(log_probs) * reward).mean()

            position_optimizer.zero_grad()
            for opt in player_optimizers.values():
                opt.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(position_model.parameters(), 1.0)
            for model in player_models.values():
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            position_optimizer.step()
            for opt in player_optimizers.values():
                opt.step()


        all_scores.append(our_score)
        all_ranks.append(our_rank)

        if (episodes_done + 1) % 50 == 0:
            window = min(100, len(all_scores))
            avg_score = np.mean(all_scores[-window:])
            avg_rank = np.mean(all_ranks[-window:])
            print(
                f"[TRAIN] Ep {episodes_done+1:4d}/{total_episodes} | "
                f"Score:{our_score:6.1f} Rank:{our_rank:2d}/12 | "
                f"Avg{window}[Sc:{avg_score:6.1f} Rk:{avg_rank:4.1f}] | "
                f"ε:{epsilon:.3f}"
            )
            print_league_roster_counts(env_train)

        if len(all_scores) >= 100:
            recent_avg = np.mean(all_scores[-100:])
            if recent_avg > best_avg_score:
                best_avg_score = recent_avg
                best_models = {
                    "position": copy.deepcopy(position_model.state_dict()),
                    "players": {
                        pos: copy.deepcopy(model.state_dict())
                        for pos, model in player_models.items()
                    },
                }

    if best_models:
        position_model.load_state_dict(best_models["position"])
        for pos, state in best_models["players"].items():
            player_models[pos].load_state_dict(state)

    torch.save(
        {
            "position": position_model.state_dict(),
            "players": {
                pos: model.state_dict() for pos, model in player_models.items()
            },
        },
        "hierarchical_agent_2023_trained.pth",
    )

    print("\n[TRAIN] ✓ Finished training on 2023")
    print(f"[TRAIN] Avg score: {np.mean(all_scores):.1f}")
    print(f"[TRAIN] Best 100-ep avg: {best_avg_score:.1f}")
    print(f"[TRAIN] Avg rank: {np.mean(all_ranks):.2f} / 12")

    return position_model, player_models, env_train, all_scores, all_ranks