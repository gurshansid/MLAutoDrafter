"""
Condense NFL player data into single row per player with weighted averages
Weights are based on recency: most recent season gets highest weight
Now includes team win percentage weighted over past 5 years (up to 2023)
"""

import pandas as pd
import numpy as np
import time

# Team name mapping - maps common abbreviations to Pro Football Reference names
TEAM_NAME_MAPPING = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB': 'Green Bay Packers',
    'GNB': 'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'JAC': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs',
    'KAN': 'Kansas City Chiefs',
    'LA': 'Los Angeles Rams',
    'LAR': 'Los Angeles Rams',
    'LAC': 'Los Angeles Chargers',
    'LV': 'Las Vegas Raiders',
    'LVR': 'Las Vegas Raiders',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots',
    'NWE': 'New England Patriots',
    'NO': 'New Orleans Saints',
    'NOR': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SF': 'San Francisco 49ers',
    'SFO': 'San Francisco 49ers',
    'SEA': 'Seattle Seahawks',
    'TB': 'Tampa Bay Buccaneers',
    'TAM': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders',
    'WSH': 'Washington Commanders',
}


def standardize_team_name(team_name):
    """
    Convert team name to Pro Football Reference format
    
    Args:
        team_name: team name from player data
    
    Returns:
        standardized team name
    """
    if pd.isna(team_name):
        return None
    
    team_name = str(team_name).strip()
    
    # If it's already a full name, return as-is
    if len(team_name) > 5:
        return team_name
    
    # Otherwise look up in mapping
    return TEAM_NAME_MAPPING.get(team_name.upper(), team_name)


def scrape_team_win_percentages(years):
    """
    Scrape team win percentages from Pro Football Reference for multiple years
    
    Args:
        years: list of years to scrape
    
    Returns:
        DataFrame with columns: year, team, wins, losses, win_pct
    """
    print(f"Scraping team win percentages for years: {years}")
    
    all_data = []
    
    for year in years:
        try:
            url = f"https://www.pro-football-reference.com/years/{year}/"
            print(f"  Fetching {year}...")
            
            # Read all tables from the page
            tables = pd.read_html(url)
            
            # AFC and NFC standings are usually tables 0 and 1
            afc = tables[0].copy()
            nfc = tables[1].copy()
            
            # Add year column
            afc['year'] = year
            nfc['year'] = year
            
            # Combine
            all_data.append(afc)
            all_data.append(nfc)
            
            # Be polite to the server
            time.sleep(1)
            
        except Exception as e:
            print(f"  ⚠️  Error fetching {year}: {e}")
            continue
    
    if not all_data:
        print("⚠️  No data scraped! Returning empty DataFrame")
        return pd.DataFrame()
    
    # Combine all years
    df = pd.concat(all_data, ignore_index=True)
    
    # Clean up column names (Pro Football Reference uses 'Tm' for team)
    if 'Tm' in df.columns:
        df = df.rename(columns={'Tm': 'team'})
    
    # Convert W and L to numeric (in case they're strings)
    df['W'] = pd.to_numeric(df['W'], errors='coerce')
    df['L'] = pd.to_numeric(df['L'], errors='coerce')
    
    # Remove rows where W or L couldn't be converted (like division headers)
    df = df.dropna(subset=['W', 'L'])
    
    # Calculate win percentage
    df['win_pct'] = df['W'] / (df['W'] + df['L'])
    
    # Keep only relevant columns
    df = df[['year', 'team', 'W', 'L', 'win_pct']]
    df.columns = ['year', 'team', 'wins', 'losses', 'win_pct']
    
    # Remove asterisks and plus signs from team names (playoff indicators)
    df['team'] = df['team'].str.replace('*', '', regex=False)
    df['team'] = df['team'].str.replace('+', '', regex=False)
    df['team'] = df['team'].str.strip()
    
    # Remove any remaining non-team rows
    df = df[df['team'].notna()]
    df = df[df['team'] != '']
    
    print(f"✅ Scraped win percentages for {len(df)} team-seasons")
    
    return df


def calculate_weighted_team_win_pct(team_records_df, team_name, last_historical_year, num_years=5):
    """
    Calculate weighted average of team's win percentage over past years
    Most recent year gets highest weight
    
    Args:
        team_records_df: DataFrame with team records
        team_name: team name to look up
        last_historical_year: the last year to include (e.g., 2023)
        num_years: number of historical years to include (default 5)
    
    Returns:
        weighted average win percentage (0.0 to 1.0)
    """
    # Standardize the team name
    standardized_team = standardize_team_name(team_name)
    
    # Get historical years (ending at last_historical_year)
    # For 2023: [2019, 2020, 2021, 2022, 2023]
    years = list(range(last_historical_year - num_years + 1, last_historical_year + 1))
    
    # Get team's records for these years
    team_history = team_records_df[
        (team_records_df['team'] == standardized_team) & 
        (team_records_df['year'].isin(years))
    ].sort_values('year')
    
    if len(team_history) == 0:
        # No history found
        return 0.500
    
    # Get win percentages (oldest to newest)
    win_pcts = team_history['win_pct'].tolist()
    seasons_available = len(win_pcts)
    
    # Create weights: 1, 2, 3, 4, 5 (where 5 is most recent)
    weights = list(range(1, seasons_available + 1))
    
    # Calculate weighted sum
    weighted_sum = sum(pct * weight for pct, weight in zip(win_pcts, weights))
    
    # Calculate sum of weights
    weight_sum = sum(weights)
    
    # Return weighted average
    weighted_avg = weighted_sum / weight_sum if weight_sum > 0 else 0.500
    
    return round(weighted_avg, 3)


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
                         target_season=2024,
                         team_history_years=5):
    """
    Condense multi-season player data into single row per player
    with weighted averages for stats BEFORE the target season
    Now includes weighted team win percentage over past years (up to 2023)
    
    Args:
        input_csv: path to input CSV file
        output_csv: path to output CSV file
        target_season: only include players who played in this season (2024),
                      but calculate stats from seasons BEFORE this
        team_history_years: number of years to use for team win % weighting (default 5)
    """
    
    print(f"Reading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Loaded {len(df)} player-season records")
    
    # Scrape team win percentages - ONLY up to 2023 (not including target_season)
    last_historical_year = target_season - 1  # 2023
    years_to_scrape = list(range(last_historical_year - team_history_years + 1, last_historical_year + 1))
    
    print(f"\nScraping team data for years: {years_to_scrape} (up to {last_historical_year})")
    
    team_records_df = scrape_team_win_percentages(years_to_scrape)
    
    if team_records_df.empty:
        print("⚠️  Warning: No team records scraped. Win percentages will default to 0.500")
    else:
        # Save team records for reference
        team_records_df.to_csv('team_win_percentages.csv', index=False)
        print(f"✅ Team win percentages saved to 'team_win_percentages.csv'")
    
    # Filter to only players who have data in the target season
    print(f"\nFiltering to players who played in {target_season}...")
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
    
    print(f"\nCondensing player data with weighted averages from historical data...")
    print(f"Team win % weighted over {team_history_years} years: {years_to_scrape[0]}-{years_to_scrape[-1]}")
    print(f"(Most recent year {years_to_scrape[-1]} gets highest weight)")
    
    # Track how many players we couldn't find team data for
    missing_team_data = 0
    
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
        
        # Get target season info (team, age, etc.)
        target_info = df[
            (df['season'] == target_season) &
            (df['first_name'] == first_name) & 
            (df['last_name'] == last_name)
        ].iloc[0]
        
        # Calculate weighted team win percentage over past years (UP TO 2023)
        team_win_pct_weighted = calculate_weighted_team_win_pct(
            team_records_df, 
            target_info['team'], 
            last_historical_year,  # 2023
            team_history_years
        )
        
        if team_win_pct_weighted == 0.500:
            missing_team_data += 1
        
        # Check if rookie (no historical data)
        is_rookie = len(player_history) == 0
        
        if is_rookie:
            # Rookie - only include ADP, all stats are 0
            player_data = {
                'first_name': first_name,
                'last_name': last_name,
                'position': position,
                'team_win_pct_weighted': team_win_pct_weighted,
                'fantasy_adp': adp_value,
                'age': target_info['age'],
                'is_rookie': 1,
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
            
            player_data = {
                'first_name': first_name,
                'last_name': last_name,
                'position': position,
                'team_win_pct_weighted': team_win_pct_weighted,
                'fantasy_adp': adp_value,
                'age': target_info['age'],
                'is_rookie': 0,
                'nfl_draft_round': most_recent_historical['nfl_draft_round'],
                'seasons_played': num_seasons,
                **weighted_stats
            }
        
        condensed_data.append(player_data)
    
    # Create DataFrame
    result_df = pd.DataFrame(condensed_data)
    
    print(f"\n⚠️  Could not find team win % data for {missing_team_data} players (defaulted to 0.500)")
    
    # Convert position to one-hot encoding (1s and 0s)
    print("\nConverting positions to one-hot encoding...")
    result_df['pos_qb'] = (result_df['position'] == 'QB').astype(int)
    result_df['pos_rb'] = (result_df['position'] == 'RB').astype(int)
    result_df['pos_wr'] = (result_df['position'] == 'WR').astype(int)
    result_df['pos_te'] = (result_df['position'] == 'TE').astype(int)
    
    # Remove original position column (no longer need team name column)
    result_df = result_df.drop('position', axis=1)
    
    # Reorder columns - NO team name, just team_win_pct_weighted
    cols = result_df.columns.tolist()
    name_cols = ['first_name', 'last_name']
    pos_cols = ['pos_qb', 'pos_rb', 'pos_wr', 'pos_te']
    team_cols = ['team_win_pct_weighted']
    other_cols = [col for col in cols if col not in name_cols + pos_cols + team_cols]
    result_df = result_df[name_cols + pos_cols + team_cols + other_cols]
    
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
    
    # Show team win percentage stats
    print(f"\nWeighted Team Win Percentage Stats (2019-2023):")
    print(f"  Mean: {result_df['team_win_pct_weighted'].mean():.3f}")
    print(f"  Min: {result_df['team_win_pct_weighted'].min():.3f}")
    print(f"  Max: {result_df['team_win_pct_weighted'].max():.3f}")
    non_default = result_df[result_df['team_win_pct_weighted'] != 0.500]
    print(f"  Non-default values: {len(non_default)} / {len(result_df)}")
    
    # Show sample of top players by position
    print("\n=== Top 5 Players by Position (Weighted PPG) ===")
    for pos_name, pos_col in [('QB', 'pos_qb'), ('RB', 'pos_rb'), ('WR', 'pos_wr'), ('TE', 'pos_te')]:
        print(f"\n{pos_name}:")
        top_pos = result_df[result_df[pos_col] == 1].head(5)[
            ['first_name', 'last_name', 'team_win_pct_weighted', 'avg_points_per_game', 'seasons_played']
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