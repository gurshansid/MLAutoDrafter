"""
Condense NFL player data into single row per player with weighted averages
Weights are based on recency: most recent season gets highest weight
"""

import pandas as pd
import numpy as np

def calculate_weighted_average(values, seasons):
    """
    Calculate weighted average where more recent seasons have higher weights
    
    Args:
        values: list of values (oldest to newest)
        seasons: number of seasons
    
    Returns:
        weighted average
    """
    if len(values) == 0 or seasons == 0:
        return 0
    
    # Create weights: 1, 2, 3, ..., n (where n is most recent)
    weights = list(range(1, seasons + 1))
    
    # Calculate weighted sum
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    
    # Calculate sum of weights (1 + 2 + 3 + ... + n)
    weight_sum = sum(weights)
    
    # Return weighted average
    return weighted_sum / weight_sum if weight_sum > 0 else 0


def condense_player_data(input_csv='nfl_player_data_with_history.csv', 
                         output_csv='nfl_players_condensed.csv',
                         target_season=2024):
    """
    Condense multi-season player data into single row per player
    with weighted averages for stats BEFORE the target season
    
    Args:
        input_csv: path to input CSV file
        output_csv: path to output CSV file
        target_season: only include players who played in this season,
                      but calculate stats from seasons BEFORE this
    """
    
    print(f"Reading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Loaded {len(df)} player-season records")
    
    # Filter to only players who have data in the target season
    print(f"Filtering to players who played in {target_season}...")
    players_in_season = df[df['season'] == target_season][['first_name', 'last_name', 'position']].drop_duplicates()
    
    # Get ADP from target season
    target_season_adp = df[df['season'] == target_season][['first_name', 'last_name', 'fantasy_adp']].copy()
    
    # Keep only historical data (BEFORE target season)
    historical_df = df[df['season'] < target_season].copy()
    
    print(f"Historical data: {len(historical_df)} player-season records before {target_season}")
    
    # Columns to calculate weighted averages for
    stat_columns = [
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
        'receiving_tds',
        'points_per_game'
    ]
    
    condensed_data = []
    
    print("Condensing player data with weighted averages from historical data...")
    
    for idx, row in players_in_season.iterrows():
        first_name = row['first_name']
        last_name = row['last_name']
        position = row['position']
        
        # Get this player's historical data
        player_history = historical_df[
            (historical_df['first_name'] == first_name) & 
            (historical_df['last_name'] == last_name)
        ].sort_values('season')
        
        # Get target season ADP
        player_adp = target_season_adp[
            (target_season_adp['first_name'] == first_name) & 
            (target_season_adp['last_name'] == last_name)
        ]
        
        adp_value = player_adp['fantasy_adp'].values[0] if len(player_adp) > 0 else 999.0
        
        # Check if rookie (no historical data)
        is_rookie = len(player_history) == 0
        
        if is_rookie:
            # Rookie - only include ADP, all stats are 0
            # Get other info from target season
            target_info = df[
                (df['season'] == target_season) &
                (df['first_name'] == first_name) & 
                (df['last_name'] == last_name)
            ].iloc[0]
            
            player_data = {
                'first_name': first_name,
                'last_name': last_name,
                'position': position,
                'team': target_info['team'],
                'fantasy_adp': adp_value,
                'age': target_info['age'],
                'is_rookie': True,
                'nfl_draft_round': target_info['nfl_draft_round'],
                'seasons_played': 0,
            }
            
            # Add all stat columns as 0
            for col in stat_columns:
                player_data[f'avg_{col}'] = 0.0
                
        else:
            # Veteran - calculate weighted averages from historical data
            num_seasons = len(player_history)
            
            # Get most recent historical data for non-averaged fields
            most_recent_historical = player_history.iloc[-1]
            
            # Calculate weighted averages for each stat
            weighted_stats = {}
            for col in stat_columns:
                if col in player_history.columns:
                    values = player_history[col].tolist()
                    weighted_stats[f'avg_{col}'] = round(
                        calculate_weighted_average(values, num_seasons), 2
                    )
                else:
                    weighted_stats[f'avg_{col}'] = 0
            
            # Get current team from target season
            target_info = df[
                (df['season'] == target_season) &
                (df['first_name'] == first_name) & 
                (df['last_name'] == last_name)
            ].iloc[0]
            
            player_data = {
                'first_name': first_name,
                'last_name': last_name,
                'position': position,
                'team': target_info['team'],
                'fantasy_adp': adp_value,
                'age': target_info['age'],
                'is_rookie': False,
                'nfl_draft_round': most_recent_historical['nfl_draft_round'],
                'seasons_played': num_seasons,
                **weighted_stats
            }
        
        condensed_data.append(player_data)
    
    # Create DataFrame
    result_df = pd.DataFrame(condensed_data)
    
    # Convert position to one-hot encoding (1s and 0s)
    print("\nConverting positions to one-hot encoding...")
    result_df['pos_qb'] = (result_df['position'] == 'QB').astype(int)
    result_df['pos_rb'] = (result_df['position'] == 'RB').astype(int)
    result_df['pos_wr'] = (result_df['position'] == 'WR').astype(int)
    result_df['pos_te'] = (result_df['position'] == 'TE').astype(int)
    
    # Remove original position column
    result_df = result_df.drop('position', axis=1)
    
    # Reorder columns to put position encoding after name
    cols = result_df.columns.tolist()
    name_cols = ['first_name', 'last_name']
    pos_cols = ['pos_qb', 'pos_rb', 'pos_wr', 'pos_te']
    other_cols = [col for col in cols if col not in name_cols + pos_cols]
    result_df = result_df[name_cols + pos_cols + other_cols]
    
    # Sort by fantasy_adp, then by weighted points per game
    result_df = result_df.sort_values(
        ['fantasy_adp', 'avg_points_per_game'], 
        ascending=[True, False]
    )
    
    # Reset index
    result_df = result_df.reset_index(drop=True)
    
    print(f"\nCondensed to {len(result_df)} unique players")
    print(f"\nPosition breakdown:")
    print(f"  QB: {result_df['pos_qb'].sum()}")
    print(f"  RB: {result_df['pos_rb'].sum()}")
    print(f"  WR: {result_df['pos_wr'].sum()}")
    print(f"  TE: {result_df['pos_te'].sum()}")
    
    # Show sample of top players by position
    print("\n=== Top 5 Players by Position (Weighted PPG) ===")
    for pos_name, pos_col in [('QB', 'pos_qb'), ('RB', 'pos_rb'), ('WR', 'pos_wr'), ('TE', 'pos_te')]:
        print(f"\n{pos_name}:")
        top_pos = result_df[result_df[pos_col] == 1].head(5)[
            ['first_name', 'last_name', 'team', 'avg_points_per_game', 'seasons_played']
        ]
        print(top_pos.to_string(index=False))
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅ Condensed data saved to '{output_csv}'")
    
    return result_df


# Example usage
if __name__ == "__main__":
    # Condense the player data
    condensed_df = condense_player_data()
    
    # Show some examples
    print("\n=== Sample Condensed Data ===")
    print(condensed_df.head(10))
    
    # Show a specific player's weighted stats
    print("\n=== Example: Find a specific player ===")
    example = condensed_df[
        (condensed_df['first_name'] == 'Christian') & 
        (condensed_df['last_name'] == 'McCaffrey')
    ]
    if not example.empty:
        print("\nChristian McCaffrey's weighted averages:")
        print(example.to_string(index=False))