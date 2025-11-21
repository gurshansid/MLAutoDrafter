"""
NFL Player Data Collection with Historical Fantasy Points
Combines Sleeper API player info with nfl-data-py for historical stats
"""

import requests
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime

class FantasyPlayerDataCollector:
    """Collect NFL player data with historical fantasy points"""
    
    SLEEPER_BASE_URL = "https://api.sleeper.app/v1"
    
    def __init__(self):
        self.sleeper_players = None
        
    def get_sleeper_players(self):
        """Fetch current player info from Sleeper"""
        print("Fetching players from Sleeper API...")
        url = f"{self.SLEEPER_BASE_URL}/players/nfl"
        response = requests.get(url)
        
        if response.status_code == 200:
            self.sleeper_players = response.json()
            print(f"Fetched {len(self.sleeper_players)} players from Sleeper")
            return self.sleeper_players
        else:
            print(f"Error fetching Sleeper data: {response.status_code}")
            return None
    
    def get_historical_stats(self, years=range(2015, 2024)):
        """
        Get historical fantasy points using nfl-data-py
        
        Args:
            years: range of years to collect (default 2015-2023)
        """
        print(f"Fetching historical stats for years {min(years)}-{max(years)}...")
        
        # Get seasonal data with fantasy points
        seasonal_data = nfl.import_seasonal_data(years)
        
        # Also need weekly data for fantasy points calculation
        weekly_data = nfl.import_weekly_data(years)
        
        # Get draft picks data (for NFL draft round info)
        print("Fetching NFL draft data...")
        try:
            draft_picks = nfl.import_draft_picks(years=range(2000, 2025))
            print(f"Loaded {len(draft_picks)} draft picks")
        except Exception as e:
            print(f"Could not fetch draft data: {e}")
            draft_picks = None
        
        # Get ADP data for each year
        print("Fetching ADP data for each year...")
        adp_data = self.get_adp_for_years(years)
        
        print(f"Collected stats for {len(seasonal_data)} player-seasons")
        
        return seasonal_data, weekly_data, draft_picks, adp_data
    
    def normalize_name(self, name):
        """
        Normalize player names for better matching
        Removes suffixes like Jr., Sr., III, etc.
        """
        if pd.isna(name) or name is None:
            return name
        
        # Convert to string and strip whitespace
        name = str(name).strip()
        
        # Remove common suffixes
        suffixes = [' Jr.', ' Jr', ' Sr.', ' Sr', ' III', ' II', ' IV', ' V']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        
        return name
    
    def calculate_age(self, birth_date):
        """Calculate age from birth date string"""
        if pd.isna(birth_date) or birth_date is None:
            return None
        try:
            birth = datetime.strptime(birth_date, '%Y-%m-%d')
            today = datetime.now()
            age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            return age
        except:
            return None
    
    def get_adp_for_years(self, years):
        """
        Fetch ADP data from Fantasy Football Calculator for multiple years
        
        Args:
            years: range of years to fetch ADP for
        
        Returns:
            Dictionary mapping (player_name, year) -> adp
        """
        import requests
        
        adp_dict = {}
        
        for year in years:
            try:
                print(f"  Fetching ADP for {year}...")
                url = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
                params = {"teams": 12, "year": year}
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                players = data.get("players", [])
                
                if not players:
                    print(f"  No ADP data for {year}")
                    continue
                
                # Store ADP for each player
                for p in players:
                    name = (
                        p.get("name") or
                        p.get("player_name") or
                        (p.get("player", {}) or {}).get("name")
                    )
                    adp = p.get("adp")
                    
                    if name and adp:
                        # Normalize name (remove suffixes)
                        name_normalized = self.normalize_name(name)
                        adp_dict[(name_normalized, year)] = adp
                
                print(f"  Loaded ADP for {len(players)} players in {year}")
                
            except Exception as e:
                print(f"  Could not fetch ADP for {year}: {e}")
                continue
        
        print(f"Total ADP entries: {len(adp_dict)}")
        return adp_dict
    
    def combine_data(self, years=range(2015, 2024)):
        """
        Combine Sleeper player info with historical fantasy points
        Returns DataFrame with: first_name, last_name, position, points, team, age, is_rookie
        """
        # Get Sleeper player data
        if self.sleeper_players is None:
            self.get_sleeper_players()
        
        # Get historical stats
        seasonal_data, weekly_data, draft_picks, adp_data = self.get_historical_stats(years)
        
        # Use weekly data which has fantasy points
        stats_df = weekly_data.copy()
        
        # Group by player and season to get total fantasy points AND detailed stats
        stats_df = stats_df.groupby(['player_id', 'season']).agg({
            'fantasy_points_ppr': 'sum',
            'week': 'count',  # Count weeks = games played
            'completions': 'sum',
            'attempts': 'sum',
            'passing_yards': 'sum',
            'passing_tds': 'sum',
            'interceptions': 'sum',
            'carries': 'sum',
            'rushing_yards': 'sum',
            'rushing_tds': 'sum',
            'receptions': 'sum',
            'targets': 'sum',
            'receiving_yards': 'sum',
            'receiving_tds': 'sum'
        }).reset_index()
        
        # Rename week count to games
        stats_df.rename(columns={'week': 'games'}, inplace=True)
        
        # Convert Sleeper data to DataFrame
        print("Processing Sleeper player data...")
        sleeper_list = []
        for player_id, player_data in self.sleeper_players.items():
            sleeper_list.append({
                'player_id': player_id,
                'first_name': player_data.get('first_name'),
                'last_name': player_data.get('last_name'),
                'position': player_data.get('position'),
                'team': player_data.get('team'),
                'birth_date': player_data.get('birth_date'),
                'years_exp': player_data.get('years_exp', 0),
                'sleeper_id': player_data.get('player_id')
            })
        
        sleeper_df = pd.DataFrame(sleeper_list)
        
        # Calculate age
        sleeper_df['age'] = sleeper_df['birth_date'].apply(self.calculate_age)
        
        # Determine if rookie (0 years experience)
        sleeper_df['is_rookie'] = sleeper_df['years_exp'] == 0
        
        # Filter to fantasy-relevant positions
        sleeper_df = sleeper_df[sleeper_df['position'].isin(['QB', 'RB', 'WR', 'TE'])]
        
        # Create full name in Sleeper data
        sleeper_df['full_name'] = sleeper_df['first_name'] + ' ' + sleeper_df['last_name']
        
        # Normalize names for better matching
        sleeper_df['full_name_normalized'] = sleeper_df['full_name'].apply(self.normalize_name)
        
        # Prepare stats data
        print("Processing historical stats...")
        print(f"Available columns in stats: {list(stats_df.columns)[:20]}")  # Debug: show columns
        
        # Import roster data to get player names and positions
        print("Fetching roster data for player names...")
        rosters = nfl.import_seasonal_rosters(years)
        
        # Merge stats with roster to get player names and positions
        stats_df = stats_df.merge(
            rosters[['player_id', 'player_name', 'position', 'team', 'season']],
            on=['player_id', 'season'],
            how='left'
        )
        
        # Filter to fantasy positions
        stats_df = stats_df[stats_df['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()
        
        # Use player_name as full_name
        stats_df['full_name'] = stats_df['player_name']
        
        # Normalize names for better matching
        stats_df['full_name_normalized'] = stats_df['full_name'].apply(self.normalize_name)
        
        # Add ADP for each player-season if available
        if adp_data:
            print("Adding ADP data by year...")
            stats_df['fantasy_adp'] = stats_df.apply(
                lambda row: adp_data.get((row['full_name_normalized'], row['season']), 999.0),
                axis=1
            )
            players_with_adp = (stats_df['fantasy_adp'] < 999).sum()
            print(f"Matched ADP for {players_with_adp} player-season records")
        else:
            print("No ADP data available")
            stats_df['fantasy_adp'] = 999.0
        
        # Add draft information if available
        if draft_picks is not None and len(draft_picks) > 0:
            print("Adding NFL draft round information...")
            
            # Create a clean draft lookup with player name
            draft_lookup = draft_picks[['pfr_player_name', 'round']].copy()
            draft_lookup['player_name_normalized'] = draft_lookup['pfr_player_name'].apply(self.normalize_name)
            
            # Remove duplicates - keep first draft (some players drafted multiple times)
            draft_lookup = draft_lookup.drop_duplicates(subset=['player_name_normalized'], keep='first')
            
            # Merge based on normalized player name
            stats_df = stats_df.merge(
                draft_lookup[['player_name_normalized', 'round']],
                left_on='full_name_normalized',
                right_on='player_name_normalized',
                how='left'
            )
            
            # Rename and clean up
            if 'round' in stats_df.columns:
                stats_df.rename(columns={'round': 'nfl_draft_round'}, inplace=True)
                # Fill missing with 0 (undrafted)
                stats_df['nfl_draft_round'] = stats_df['nfl_draft_round'].fillna(0)
                print(f"Draft round data added. Players with draft info: {(stats_df['nfl_draft_round'] > 0).sum()}")
            
            if 'player_name_normalized' in stats_df.columns:
                stats_df.drop('player_name_normalized', axis=1, inplace=True)
        else:
            print("No draft data available, skipping draft round")
        
        # Merge the datasets
        print("Merging Sleeper and stats data...")
        
        # Use LEFT join from Sleeper data to keep ALL players (including rookies)
        # Merge on normalized names for better matching (handles Jr., Sr., etc.)
        stats_columns = ['full_name_normalized', 'season', 'fantasy_points_ppr', 'games',
                        'team', 'position', 'completions', 'attempts', 'passing_yards',
                        'passing_tds', 'interceptions', 'carries', 'rushing_yards',
                        'rushing_tds', 'receptions', 'targets', 'receiving_yards', 'receiving_tds',
                        'nfl_draft_round', 'fantasy_adp']
        
        # Only use columns that exist
        available_stats_columns = [col for col in stats_columns if col in stats_df.columns]
        
        combined = sleeper_df.merge(
            stats_df[available_stats_columns],
            on='full_name_normalized',
            how='left',
            suffixes=('', '_stats')
        )
        
        # For players without stats (rookies), fill season with current year
        current_year = datetime.now().year
        combined['season'] = combined['season'].fillna(current_year)
        
        # Fill NaN fantasy points and games with 0 for rookies/players without history
        numeric_columns = ['fantasy_points_ppr', 'games', 'completions', 'attempts',
                          'passing_yards', 'passing_tds', 'interceptions', 'carries',
                          'rushing_yards', 'rushing_tds', 'receptions', 'targets',
                          'receiving_yards', 'receiving_tds', 'nfl_draft_round', 'fantasy_adp']
        
        for col in numeric_columns:
            if col in combined.columns:
                combined[col] = combined[col].fillna(0)
        
        # Use Sleeper team if stats team is missing
        if 'team_stats' in combined.columns:
            combined['team'] = combined['team_stats'].fillna(combined['team'])
            combined.drop('team_stats', axis=1, inplace=True)
        
        # Use Sleeper position if stats position is missing
        if 'position_stats' in combined.columns:
            combined['position'] = combined['position_stats'].fillna(combined['position'])
            combined.drop('position_stats', axis=1, inplace=True)
        
        # Remove duplicates - keep first occurrence of each player-season
        combined = combined.drop_duplicates(subset=['player_id', 'season'], keep='first')
        
        # Select final columns
        final_columns = [
            'first_name',
            'last_name', 
            'position',
            'team',
            'season',
            'fantasy_points_ppr',
            'points_per_game',
            'fantasy_adp',
            'age',
            'is_rookie',
            'nfl_draft_round',
            'games',
            'completions',
            'attempts',
            'passing_yards',
            'passing_tds',
            'interceptions',
            'carries',
            'rushing_yards',
            'rushing_tds',
            'receptions',
            'targets',
            'receiving_yards', 
            'receiving_tds'
        ]
        
        # Only keep columns that exist
        available_columns = [col for col in final_columns if col in combined.columns]
        result = combined[available_columns].copy()
        
        # Don't drop rows with missing fantasy points anymore (keep rookies with 0 points)
        result = result.dropna(subset=['first_name', 'last_name'])
        
        # Calculate points per game
        result['points_per_game'] = result.apply(
            lambda row: row['fantasy_points_ppr'] / row['games'] if row['games'] > 0 else 0,
            axis=1
        )
        result['points_per_game'] = result['points_per_game'].round(2)
        
        # Fill NaN teams with 'FA' (Free Agent) temporarily
        result['team'] = result['team'].fillna('FA')
        
        # Remove free agents completely
        result = result[result['team'] != 'FA']
        
        # Sort by team, then season, then fantasy points
        result = result.sort_values(['first_name', 'last_name', 'season'], ascending=[True, True, False])
        
        # Reset index for clean output
        result = result.reset_index(drop=True)
        
        print(f"\nFinal dataset: {len(result)} player-season records")
        print(f"Unique players: {result[['first_name', 'last_name']].drop_duplicates().shape[0]}")
        
        return result


# Example usage
if __name__ == "__main__":
    # Install required packages first:
    # pip install requests pandas nfl-data-py
    
    collector = FantasyPlayerDataCollector()
    
    # Get combined data for last 6 years (2019-2024)
    df = collector.combine_data(years=range(2019, 2025))
    
    print("\n=== Sample Data ===")
    print(df.head(10))
    
    print("\n=== Data Summary ===")
    print(f"Total records: {len(df)}")
    print(f"\nPosition breakdown:")
    print(df['position'].value_counts())
    
    print(f"\nSeasons covered:")
    print(df['season'].value_counts().sort_index())
    
    print(f"\nRookie breakdown:")
    print(df['is_rookie'].value_counts())
    
    # Show top fantasy performers
    print("\n=== Top 10 Fantasy Performers (Single Season) ===")
    top_performers = df.nlargest(10, 'fantasy_points_ppr')[
        ['first_name', 'last_name', 'position', 'season', 'fantasy_points_ppr', 'team']
    ]
    print(top_performers)
    
    # Save to CSV
    df.to_csv('nfl_player_data_with_history.csv', index=False)
    print("\n✅ Data saved to 'nfl_player_data_with_history.csv'")