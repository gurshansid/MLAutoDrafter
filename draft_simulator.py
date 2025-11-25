"""
Fantasy Football Draft Simulator - FIXED to use real ADP
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Optional

class DraftSimulator:
    """Simulates a fantasy football draft with roster limits"""
    
    def __init__(self, 
                 player_data_path: str,
                 n_teams: int = 12,
                 n_rounds: int = 9):
        """
        Initialize draft simulator
        
        Args:
            player_data_path: Path to CSV with player data
            n_teams: Number of teams in draft
            n_rounds: Total rounds in draft (9 for standard roster)
        """
        self.player_data = pd.read_csv(player_data_path)
        self.n_teams = n_teams
        self.n_rounds = n_rounds
        
        # Roster limits
        self.roster_limits = {
            'QB': 1,
            'RB': 4,
            'WR': 4,
            'TE': 3,
            'K': 1
        }
        
        self.roster_minimums = {
            'QB': 1,
            'RB': 2,
            'WR': 2,
            'TE': 1,
            'K': 1
        }
        
        # Initialize draft state
        self.reset_draft()
        
    def reset_draft(self):
        """Reset draft to initial state"""
        self.available_players = self._prepare_player_pool()
        
        self.rosters = {i: [] for i in range(self.n_teams)}
        self.roster_counts = {i: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'K': 0} 
                              for i in range(self.n_teams)}
        
        self.current_round = 1
        self.current_pick = 0
        self.draft_history = []
        
    def _prepare_player_pool(self):
        """Prepare available players with REAL pre-season ADP"""
        # Get most recent season for each player
        df = self.player_data.sort_values('season', ascending=False)
        df = df.drop_duplicates(subset=['first_name', 'last_name'], keep='first')
        
        # Filter out players with 0 points (unless rookies)
        df = df[(df['fantasy_points_ppr'] > 0) | (df['is_rookie'] == True)]
        
        # CRITICAL FIX: Use REAL fantasy_adp if available
        if 'fantasy_adp' in df.columns:
            # For players with real ADP (< 999), use it
            # For players without ADP, assign based on points but push them down
            df['adp'] = df['fantasy_adp'].copy()
            
            # Players without real ADP get ranked by points but start after ADP 200
            no_adp_mask = df['fantasy_adp'] >= 999
            if no_adp_mask.sum() > 0:
                # Sort players without ADP by fantasy points
                no_adp_players = df[no_adp_mask].sort_values('fantasy_points_ppr', ascending=False)
                # Assign them ADP starting from 200
                df.loc[no_adp_mask, 'adp'] = 200 + np.arange(no_adp_mask.sum())
        else:
            # Fallback if no ADP column exists
            df = df.sort_values('fantasy_points_ppr', ascending=False)
            df['adp'] = range(1, len(df) + 1)
        
        # Add some randomness to rookies' ADP (they're uncertain)
        rookie_mask = df['is_rookie'] == True
        if rookie_mask.sum() > 0:
            df.loc[rookie_mask, 'adp'] = df.loc[rookie_mask, 'adp'].apply(
                lambda x: x + random.randint(-20, 50) if x < 200 else x + random.randint(20, 100)
            )
        
        # Add placeholder kickers if not in data
        if 'K' not in df['position'].values or df[df['position'] == 'K'].shape[0] < 12:
            kickers_data = []
            kicker_names = ['Justin Tucker', 'Harrison Butker', 'Tyler Bass', 'Evan McPherson', 
                           'Daniel Carlson', 'Jason Sanders', 'Jake Elliott', 'Brandon McManus',
                           'Greg Zuerlein', 'Matt Gay', 'Cairo Santos', 'Nick Folk']
            for i, name in enumerate(kicker_names):
                first, last = name.split()
                kickers_data.append({
                    'first_name': first,
                    'last_name': last,
                    'position': 'K',
                    'team': 'FA',
                    'fantasy_points_ppr': 145 - (i * 3),  # Kickers score ~145 pts
                    'is_rookie': False,
                    'age': 28,
                    'adp': 150 + i * 5  # Kickers drafted late
                })
            kickers_df = pd.DataFrame(kickers_data)
            df = pd.concat([df, kickers_df], ignore_index=True)
        
        return df.reset_index(drop=True)
    
    def get_draft_order(self, round_num: int) -> List[int]:
        """Get draft order for a given round (snake draft)"""
        if round_num % 2 == 1:
            return list(range(self.n_teams))
        else:
            return list(range(self.n_teams - 1, -1, -1))
    
    def get_available_players(self) -> pd.DataFrame:
        """Get currently available players"""
        drafted_indices = [p['player_index'] for p in self.draft_history]
        return self.available_players[~self.available_players.index.isin(drafted_indices)]
    
    def can_draft_position(self, team_id: int, position: str) -> bool:
        """Check if team can draft a player at this position"""
        current_count = self.roster_counts[team_id].get(position, 0)
        max_allowed = self.roster_limits.get(position, 0)
        return current_count < max_allowed
    
    def get_team_needs(self, team_id: int) -> Dict[str, int]:
        """Get positions the team still needs (priority levels)"""
        needs = {}
        counts = self.roster_counts[team_id]
        picks_remaining = self.n_rounds - len([p for p in self.draft_history if p['team'] == team_id])
        
        total_skill_players = counts['RB'] + counts['WR'] + counts['TE']
        skill_players_needed = 8 - counts['QB'] - counts['K'] - total_skill_players
        
        # Priority 3: Must draft (minimums not met)
        if counts['QB'] < 1 and picks_remaining <= (1 - counts['QB']):
            needs['QB'] = 3
        if counts['K'] < 1 and picks_remaining <= (1 - counts['K']):
            needs['K'] = 3
            
        min_still_needed = max(0, 2 - counts['RB']) + max(0, 2 - counts['WR']) + max(0, 1 - counts['TE'])
        
        if counts['RB'] < 2:
            needs['RB'] = 3 if picks_remaining <= min_still_needed else 2
        if counts['WR'] < 2:
            needs['WR'] = 3 if picks_remaining <= min_still_needed else 2
        if counts['TE'] < 1:
            needs['TE'] = 3 if picks_remaining <= min_still_needed + 1 else 1
        
        # Priority 1: Nice to have (for flex spots)
        if 'RB' not in needs and counts['RB'] < 4:
            needs['RB'] = 1
        if 'WR' not in needs and counts['WR'] < 4:
            needs['WR'] = 1
        if 'TE' not in needs and counts['TE'] < 3:
            needs['TE'] = 1
            
        return needs
    
    def draft_player_by_adp(self, team_id: int, randomness: float = 0.2):
        """Draft a player using ADP with roster limits"""
        available = self.get_available_players()
        
        if len(available) == 0:
            return None
        
        # Filter for positions the team can still draft
        can_draft = []
        for _, player in available.iterrows():
            if self.can_draft_position(team_id, player['position']):
                can_draft.append(player)
        
        if not can_draft:
            print(f"  Team {team_id} has no valid positions to draft!")
            return None
            
        available_filtered = pd.DataFrame(can_draft)
        available_filtered = available_filtered.sort_values('adp')
        
        # Get team needs
        team_needs = self.get_team_needs(team_id)
        
        # Adjust consideration pool based on round
        if self.current_round <= 3:
            n_consider = min(5, len(available_filtered))
        elif self.current_round <= 6:
            n_consider = min(8, len(available_filtered))
        else:
            n_consider = min(12, len(available_filtered))
        
        candidates = available_filtered.head(n_consider).copy()
        
        # Calculate weights
        weights = []
        for idx, player in candidates.iterrows():
            base_weight = np.exp(-0.3 * list(candidates.index).index(idx))
            
            position_need = team_needs.get(player['position'], 0)
            if position_need == 3:
                base_weight *= 10
            elif position_need == 2:
                base_weight *= 3
            elif position_need == 1:
                base_weight *= 1.2
            
            # Kicker penalty in early rounds
            if player['position'] == 'K' and self.current_round <= 7:
                base_weight *= 0.1
                
            weights.append(base_weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        selected_idx = random.choices(list(candidates.index), weights=weights, k=1)[0]
        return available_filtered.loc[selected_idx]
    
    def draft_player_by_model(self, team_id: int, model_predictions: np.ndarray):
        """Draft a player using model predictions with roster limits"""
        available = self.get_available_players()
        
        if len(available) == 0:
            return None
        
        # Filter for valid positions
        valid_indices = []
        valid_predictions = []
        
        for i, (idx, player) in enumerate(available.iterrows()):
            if self.can_draft_position(team_id, player['position']):
                valid_indices.append(idx)
                valid_predictions.append(model_predictions[i])
        
        if not valid_indices:
            return None
        
        # Renormalize predictions
        valid_predictions = np.array(valid_predictions)
        valid_predictions = valid_predictions / valid_predictions.sum()
        
        selected_idx = np.random.choice(valid_indices, p=valid_predictions)
        return available.loc[selected_idx]
    
    def make_pick(self, team_id: int, player: pd.Series):
        """Record a draft pick"""
        pick_info = {
            'round': self.current_round,
            'pick': len(self.draft_history) + 1,
            'team': team_id,
            'player_index': player.name,
            'player_name': f"{player['first_name']} {player['last_name']}",
            'position': player['position'],
            'fantasy_points': player['fantasy_points_ppr']
        }
        
        self.draft_history.append(pick_info)
        self.rosters[team_id].append(player)
        
        position = player['position']
        if position in self.roster_counts[team_id]:
            self.roster_counts[team_id][position] += 1
    
    def simulate_draft(self, our_team_id=0, our_model=None):
        """Simulate a complete draft"""
        self.reset_draft()
        
        for round_num in range(1, self.n_rounds + 1):
            self.current_round = round_num
            draft_order = self.get_draft_order(round_num)
            
            for team_id in draft_order:
                if team_id == our_team_id and our_model is not None:
                    available = self.get_available_players()
                    if len(available) > 0:
                        predictions = np.ones(len(available)) / len(available)
                        player = self.draft_player_by_model(team_id, predictions)
                else:
                    player = self.draft_player_by_adp(team_id)
                
                if player is not None:
                    self.make_pick(team_id, player)
        
        return self.draft_history, self.rosters
    
    def get_starting_lineup(self, team_id: int) -> Dict[str, List]:
        """Determine optimal starting lineup"""
        roster = self.rosters[team_id]
        lineup = {
            'QB': [], 'RB': [], 'WR': [], 'TE': [], 'FLEX': [], 'K': [], 'bench': []
        }
        
        assigned_indices = set()
        
        qbs = sorted([p for p in roster if p['position'] == 'QB'], 
                    key=lambda x: x['fantasy_points_ppr'], reverse=True)
        rbs = sorted([p for p in roster if p['position'] == 'RB'],
                    key=lambda x: x['fantasy_points_ppr'], reverse=True)
        wrs = sorted([p for p in roster if p['position'] == 'WR'],
                    key=lambda x: x['fantasy_points_ppr'], reverse=True)
        tes = sorted([p for p in roster if p['position'] == 'TE'],
                    key=lambda x: x['fantasy_points_ppr'], reverse=True)
        ks = sorted([p for p in roster if p['position'] == 'K'],
                    key=lambda x: x['fantasy_points_ppr'], reverse=True)
        
        if qbs: 
            lineup['QB'] = qbs[:1]
            for p in qbs[:1]:
                assigned_indices.add(p.name)
        if rbs: 
            lineup['RB'] = rbs[:2]
            for p in rbs[:2]:
                assigned_indices.add(p.name)
        if wrs: 
            lineup['WR'] = wrs[:2]
            for p in wrs[:2]:
                assigned_indices.add(p.name)
        if tes: 
            lineup['TE'] = tes[:1]
            for p in tes[:1]:
                assigned_indices.add(p.name)
        if ks: 
            lineup['K'] = ks[:1]
            for p in ks[:1]:
                assigned_indices.add(p.name)
        
        flex_candidates = []
        if len(rbs) > 2: flex_candidates.extend(rbs[2:])
        if len(wrs) > 2: flex_candidates.extend(wrs[2:])
        if len(tes) > 1: flex_candidates.extend(tes[1:])
        
        flex_candidates.sort(key=lambda x: x['fantasy_points_ppr'], reverse=True)
        lineup['FLEX'] = flex_candidates[:2]
        for p in flex_candidates[:2]:
            assigned_indices.add(p.name)
        
        lineup['bench'] = [p for p in roster if p.name not in assigned_indices]
        
        return lineup
    
    def calculate_roster_score(self, team_id: int) -> float:
        """Calculate fantasy points for starting lineup only"""
        lineup = self.get_starting_lineup(team_id)
        
        total = 0
        for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX', 'K']:
            for player in lineup[pos]:
                total += player['fantasy_points_ppr']
        
        return total
    
    def print_roster_summary(self, team_id: int):
        """Print a nice summary of a team's roster"""
        lineup = self.get_starting_lineup(team_id)
        counts = self.roster_counts[team_id]
        
        print(f"\n=== Team {team_id} Roster ===")
        print("STARTERS:")
        
        for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX', 'K']:
            if lineup[pos]:
                print(f"{pos}:")
                for player in lineup[pos]:
                    print(f"  {player['first_name']} {player['last_name']} ({player['position']}) - {player['fantasy_points_ppr']:.1f} pts")
        
        if lineup['bench']:
            print("\nBENCH:")
            for player in lineup['bench']:
                print(f"  {player['first_name']} {player['last_name']} ({player['position']}) - {player['fantasy_points_ppr']:.1f} pts")
        
        print(f"\nPosition counts: QB:{counts['QB']} RB:{counts['RB']} WR:{counts['WR']} TE:{counts['TE']} K:{counts['K']}")
        print(f"Starting lineup points: {self.calculate_roster_score(team_id):.1f}")
        
        issues = []
        if counts['QB'] < 1: issues.append("Need QB")
        if counts['RB'] < 2: issues.append(f"Need {2-counts['RB']} RB")
        if counts['WR'] < 2: issues.append(f"Need {2-counts['WR']} WR")
        if counts['TE'] < 1: issues.append("Need TE")
        if counts['K'] < 1: issues.append("Need K")
        
        if issues:
            print(f"⚠️  Invalid roster: {', '.join(issues)}")
        else:
            print("✅ Valid roster!")
            
    def evaluate_draft(self) -> Dict[int, float]:
        """Evaluate all teams' draft results"""
        scores = {}
        for team_id in range(self.n_teams):
            scores[team_id] = self.calculate_roster_score(team_id)
        return scores