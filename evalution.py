import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from envirement import DraftEnvironment
from nn import HierarchicalAgent
from features import get_qb_features, get_rb_features, get_wr_features, get_te_features




def plot_training(scores, ranks, filename: str = "hierarchical_training_2023.png"):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(scores, alpha=0.3, linewidth=0.5)
    if len(scores) >= 100:
        ma = pd.Series(scores).rolling(100).mean()
        ax.plot(ma, linewidth=2.5, label="100-Ep MA")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score (2023)")
    ax.set_title("Training Scores on 2023")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ranks, alpha=0.3, linewidth=0.5)
    if len(ranks) >= 100:
        ma = pd.Series(ranks).rolling(100).mean()
        ax.plot(ma, linewidth=2.5, label="100-Ep MA")
        ax.axhline(y=6.5, linestyle="--", label="Random")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rank (1=Best)")
    ax.set_title("Training Ranks on 2023")
    ax.set_ylim(12.5, 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"[PLOT] ✓ Saved training plot to {filename}")
    plt.close()


def evaluate_on_2024(
    position_model,
    player_models,
    condensed_path: str = "nfl_players_condensed.csv",
    stats_2024_path: str = "nfl_players_2024_stats.csv",
    n_eval_drafts: int = 500,
):


    device = next(position_model.parameters()).device
    env_eval = DraftEnvironment(
        condensed_path=condensed_path,
        stats_path=stats_2024_path,
        n_teams=12,
        n_rounds=8,
        seed=123,
    )

    agent_eval = HierarchicalAgent(position_model, player_models, env_eval)

    scores_2024 = []
    ranks_2024 = []

    print("\n[EVAL] Evaluating frozen 2023-trained policy on 2024...\n")

    for ep in range(n_eval_drafts):
        our_team = env_eval.reset()

        for rnd in range(1, env_eval.n_rounds + 1):
            env_eval.current_round = rnd
            order = env_eval.get_draft_order(rnd)

            for tid in order:
                if tid == our_team:
                    # Pure exploitation: epsilon=0, training=False
                    player, _ = agent_eval.pick_player(
                        env_eval, tid, rnd, epsilon=0.0, training=False
                    )
                    if player is not None:
                        env_eval.make_pick(tid, player)
                else:
                    player = env_eval.bot_pick(tid)
                    if player is not None:
                        env_eval.make_pick(tid, player)

        scores_dict = env_eval.evaluate_league()
        our_score = scores_dict[our_team]
        league_scores = list(scores_dict.values())
        our_rank = sorted(league_scores, reverse=True).index(our_score) + 1

        scores_2024.append(our_score)
        ranks_2024.append(our_rank)

        if (ep + 1) % 50 == 0:
            print(
                f"[EVAL] Draft {ep+1:4d}/{n_eval_drafts} | "
                f"Score_2024:{our_score:6.1f} Rank_2024:{our_rank:2d}/12"
            )

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY – 2023-TRAINED POLICY ON 2024")
    print("=" * 80)
    print(f"Drafts evaluated: {n_eval_drafts}")
    print(f"Avg 2024 score: {np.mean(scores_2024):.1f}")
    print(f"Std 2024 score: {np.std(scores_2024):.1f}")
    print(f"Best 2024 score: {np.max(scores_2024):.1f}")
    print(f"Worst 2024 score: {np.min(scores_2024):.1f}")
    print(f"Avg 2024 rank:  {np.mean(ranks_2024):.2f} / 12")
    print(f"Best 2024 rank: {np.min(ranks_2024)}")
    print(f"Worst 2024 rank:{np.max(ranks_2024)}")
    print("=" * 80 + "\n")

    return scores_2024, ranks_2024


def inspect_learned_policy(position_model, player_models, env: DraftEnvironment):

    device = next(position_model.parameters()).device
    position_model.eval()
    for m in player_models.values():
        m.eval()

    positions = ["QB", "RB", "WR", "TE"]

    print()
    print("=" * 80)
    print("HIERARCHICAL RL - Position + Player Policies")
    print("=" * 80)
    print()

    mid_team = env.n_teams // 2
    env.reset(our_team_id=mid_team)

    def eval_player_value(row, pos: str):
        if pos == "QB":
            feats = get_qb_features(row, env.feature_means, env.feature_stds)
        elif pos == "RB":
            feats = get_rb_features(row, env.feature_means, env.feature_stds)
        elif pos == "WR":
            feats = get_wr_features(row, env.feature_means, env.feature_stds)
        else:
            feats = get_te_features(row, env.feature_means, env.feature_stds)

        feats_t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        context = env.get_context_vector(mid_team, round_num=min(4, env.n_rounds))
        ctx_t = torch.tensor(
            context, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            val = player_models[pos](feats_t, ctx_t).item()
        return val

    for pos in positions:
        subset = env.player_pool[env.player_pool["position"] == pos]
        if subset.empty:
            continue

        player_vals = []
        for _, row in subset.iterrows():
            val = eval_player_value(row, pos)
            player_vals.append((row, val))

        player_vals.sort(key=lambda x: x[1], reverse=True)

        print(f"Top 10 {pos}s according to the learned player policy:")
        for rank, (row, val) in enumerate(player_vals[:10], start=1):
            name = f"{row['first_name']} {row['last_name']}"
            adp = row["fantasy_adp"]
            print(f"  {rank:2d}. {name:<24} value={val:6.3f}  ADP={adp:6.1f}")
        print()