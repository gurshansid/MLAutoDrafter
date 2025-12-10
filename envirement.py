#Envirement + drafting rules

import random
from typing import Optional, Dict

import numpy as np
import pandas as pd


class DraftEnvironment:


    def __init__(
        self,
        condensed_path: str,
        stats_path: str,
        n_teams: int = 12,
        n_rounds: int = 8,
        seed: int = 42,
    ):
        self.n_teams = n_teams
        self.n_rounds = n_rounds
        self.rng = random.Random(seed)

        # Hard roster caps:
        self.roster_limits = {"QB": 1, "RB": 5, "WR": 5, "TE": 3}

        
        self.min_requirements = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
        self.max_players_per_team = 8

        # For context features / "needs"
        self.starting_requirements = {"QB": 1, "RB": 2, "WR": 2, "FLEX": 2, "TE": 1}


        df_condensed = pd.read_csv(condensed_path)


        if "pos_qb" in df_condensed.columns:
            df_condensed["position"] = "UNKNOWN"
            df_condensed.loc[df_condensed["pos_qb"] == 1, "position"] = "QB"
            df_condensed.loc[df_condensed["pos_rb"] == 1, "position"] = "RB"
            df_condensed.loc[df_condensed["pos_wr"] == 1, "position"] = "WR"
            df_condensed.loc[df_condensed["pos_te"] == 1, "position"] = "TE"

 
        df_stats = pd.read_csv(stats_path)

        if "fantasy_points_ppr" not in df_stats.columns:
            raise ValueError(f"{stats_path} must contain fantasy_points_ppr")

        df = df_condensed.merge(
            df_stats[["first_name", "last_name", "fantasy_points_ppr"]],
            on=["first_name", "last_name"],
            how="inner", 
        )

        df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
        if "fantasy_adp" in df.columns:
            df = df[df["fantasy_adp"] < 300].copy()

        df = df.sort_values("fantasy_adp").reset_index(drop=True)
        self.player_pool = df


        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if not c.startswith("fantasy_points_ppr")]

        self.feature_means = df[numeric_cols].mean()
        self.feature_stds = df[numeric_cols].std().replace(0, 1.0)

        self.reset()
        print(f"[Environment] Loaded {len(df)} players from {stats_path}")
        print(f"[Environment] Positions: {df['position'].value_counts().to_dict()}")

   
    #Draft mechanics
    def reset(self, our_team_id: Optional[int] = None):
        """Reset the draft state"""
        self.our_team_id = (
            our_team_id
            if our_team_id is not None
            else self.rng.randint(0, self.n_teams - 1)
        )
        self.current_round = 1
        self.draft_history = []
        self.rosters: Dict[int, list] = {tid: [] for tid in range(self.n_teams)}
        self.roster_counts: Dict[int, Dict[str, int]] = {
            tid: {pos: 0 for pos in self.roster_limits.keys()}
            for tid in range(self.n_teams)
        }
        return self.our_team_id

    def get_draft_order(self, round_num: int):
        return (
            list(range(self.n_teams))
            if round_num % 2 == 1
            else list(range(self.n_teams - 1, -1, -1))
        )

    def get_available_players(self):
        taken = {p["player_index"] for p in self.draft_history}
        return self.player_pool[~self.player_pool.index.isin(taken)]


    def get_roster_needs(self, team_id: int):

        needs = {}
        for pos in ["QB", "RB", "WR", "TE"]:
            needs[pos] = max(
                0, self.starting_requirements[pos] - self.roster_counts[team_id][pos]
            )

        total_needed = self.max_players_per_team
        filled = sum(self.roster_counts[team_id].values())
        needs["FLEX"] = max(0, total_needed - filled)
        return needs

    def position_is_legal_for_roster(self, team_id: int, position: str) -> bool:

        counts = self.roster_counts[team_id]
        total = sum(counts.values())


        if total >= self.max_players_per_team:
            return False

        new_counts = counts.copy()
        new_counts[position] += 1
        new_total = total + 1
        remaining_picks = self.max_players_per_team - new_total

        if new_counts[position] > self.roster_limits[position]:
            return False

        needed = 0
        for pos, req in self.min_requirements.items():
            if new_counts[pos] < req:
                needed += (req - new_counts[pos])

        if needed > remaining_picks:
            return False

        return True

    def can_draft_position(self, team_id: int, position: str):
        return self.position_is_legal_for_roster(team_id, position)

    def get_context_vector(self, team_id: int, round_num: int):
        needs = self.get_roster_needs(team_id)
        roster_size = len(self.rosters[team_id])
        picks_remaining = self.n_rounds - round_num + 1
        draft_position = (
            team_id / (self.n_teams - 1) if self.n_teams > 1 else 0.5
        )
        order = self.get_draft_order(round_num)
        pick_in_round = order.index(team_id) + 1

        return np.array(
            [
                round_num / self.n_rounds,
                pick_in_round / self.n_teams,
                needs.get("QB", 0),
                needs.get("RB", 0),
                needs.get("WR", 0),
                needs.get("TE", 0),
                needs.get("FLEX", 0),
                roster_size / self.n_rounds,
                picks_remaining / self.n_rounds,
                draft_position,
            ],
            dtype=np.float32,
        )

    def get_available_at_position(self, team_id: int, position: str):
        if not self.position_is_legal_for_roster(team_id, position):
            return []

        available = self.get_available_players()
        if available.empty:
            return []

        avail_pos = available[available["position"] == position]
        return [row for _, row in avail_pos.iterrows()]

    def make_pick(self, team_id: int, player_row: pd.Series):
        info = {
            "round": self.current_round,
            "team_id": team_id,
            "player_index": player_row.name,
            "position": player_row["position"],
            "first_name": player_row["first_name"],
            "last_name": player_row["last_name"],
            "fantasy_points_ppr": player_row["fantasy_points_ppr"],
            "adp": player_row["fantasy_adp"],
        }
        self.draft_history.append(info)
        self.rosters[team_id].append(player_row)
        if player_row["position"] in self.roster_counts[team_id]:
            self.roster_counts[team_id][player_row["position"]] += 1

    def bot_pick(self, team_id: int, slippage: int = 15):
        if len(self.rosters[team_id]) >= self.max_players_per_team:
            return None

        available = self.get_available_players()
        if available.empty:
            return None

        legal_rows = []
        for _, row in available.iterrows():
            pos = row["position"]
            if self.position_is_legal_for_roster(team_id, pos):
                legal_rows.append(row)

        if not legal_rows:
            return None

        legal_df = pd.DataFrame(legal_rows).sort_values("fantasy_adp")

        top_n = min(slippage, len(legal_df))
        top_players = legal_df.iloc[:top_n]

        weights = np.zeros(top_n, dtype=np.float64)

        if top_n >= 1:
            weights[0] = 0.85  # best ADP
        if top_n >= 2:
            weights[1] = 0.10  # second best
        if top_n >= 3:
            weights[2] = 0.04  # third best

        if top_n > 3:
            remaining_prob = 1.0 - weights.sum()
            remaining_prob = max(remaining_prob, 0.0)
            tail_count = top_n - 3
            if tail_count > 0 and remaining_prob > 0:
                weights[3:] = remaining_prob / tail_count

        if weights.sum() == 0:
            weights[:] = 1.0 / top_n
        else:
            weights /= weights.sum()

        chosen_idx = np.random.choice(top_n, p=weights)
        return top_players.iloc[chosen_idx]

    def build_starting_lineup(self, team_id: int):

        players = self.rosters[team_id]
        lineup = {"QB": [], "RB": [], "WR": [], "TE": [], "FLEX": [], "BENCH": []}

        by_pos = {pos: [] for pos in self.roster_limits.keys()}
        for p in players:
            if p["position"] in by_pos:
                by_pos[p["position"]].append(p)

        for pos in by_pos:
            by_pos[pos].sort(key=lambda x: x["fantasy_points_ppr"], reverse=True)

        used = set()

        if by_pos["QB"]:
            lineup["QB"].append(by_pos["QB"][0])
            used.add(by_pos["QB"][0].name)

        for pos, n in [("RB", 2), ("WR", 2), ("TE", 1)]:
            for p in by_pos[pos][:n]:
                lineup[pos].append(p)
                used.add(p.name)

        flex_cands = [
            p
            for pos in ["RB", "WR", "TE"]
            for p in by_pos[pos]
            if p.name not in used
        ]
        flex_cands.sort(key=lambda x: x["fantasy_points_ppr"], reverse=True)
        for p in flex_cands[:2]:
            lineup["FLEX"].append(p)
            used.add(p.name)

        for p in players:
            if p.name not in used:
                lineup["BENCH"].append(p)

        return lineup

    def team_score(self, team_id: int):
        """Calculate total fantasy points for a team"""
        lineup = self.build_starting_lineup(team_id)
        return sum(
            float(p["fantasy_points_ppr"])
            for pos in ["QB", "RB", "WR", "TE", "FLEX"]
            for p in lineup[pos]
        )

    def evaluate_league(self):
        """Evaluate all teams' scores"""
        return {tid: self.team_score(tid) for tid in range(self.n_teams)}



def print_agent_roster(env: DraftEnvironment, team_id: int):
    """Pretty-print the agent's roster for the current draft (all starters)."""
    lineup = env.build_starting_lineup(team_id)

    print("  Starters:")
    for pos in ["QB", "RB", "WR", "TE", "FLEX"]:
        for p in lineup[pos]:
            name = f"{p['first_name']} {p['last_name']}"
            pts = float(p["fantasy_points_ppr"])
            adp = float(p["fantasy_adp"])
            print(f"    {pos:<5} {name:<25} PPR={pts:6.1f}  ADP={adp:6.1f}")

    if lineup["BENCH"]:
        print("  Bench:")
        for p in lineup["BENCH"]:
            name = f"{p['first_name']} {p['last_name']}"
            pts = float(p["fantasy_points_ppr"])
            adp = float(p["fantasy_adp"])
            print(f"    BENCH {name:<25} PPR={pts:6.1f}  ADP={adp:6.1f}")


def v(env: DraftEnvironment):
    print("  League roster counts (current draft):")
    print("    Team |  QB  RB  WR  TE | Total")
    print("    ------------------------------")
    for tid in range(env.n_teams):
        counts = env.roster_counts[tid]
        total = sum(counts.values())
        print(
            f"     {tid:2d} |  {counts['QB']:2d} {counts['RB']:3d} "
            f"{counts['WR']:3d} {counts['TE']:3d} | {total:4d}"
        )
    print()