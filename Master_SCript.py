"""
Master script to configure and run all NFL fantasy data collection
Change seasons in one place and run everything
"""

# ============================================================================
# CONFIGURATION - CHANGE THESE VALUES
# ============================================================================

# Years to collect historical data for (will exclude the target season from final dataset)
HISTORICAL_YEARS = range(2019, 2025)  # 2019-2024

# Target season for predictions (stats calculated from seasons BEFORE this)
TARGET_SEASON = 2024

# ============================================================================
# END CONFIGURATION
# ============================================================================

import sys
import importlib.util

def load_module_from_file(module_name, file_path):
    """Load a Python module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main():
    print("=" * 80)
    print("NFL FANTASY DATA COLLECTION PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Historical Years: {min(HISTORICAL_YEARS)}-{max(HISTORICAL_YEARS)}")
    print(f"  Target Season: {TARGET_SEASON}")
    print(f"  Prediction Stats: Based on seasons {min(HISTORICAL_YEARS)}-{TARGET_SEASON-1}")
    print(f"  Actual Stats: From season {TARGET_SEASON}")
    print("\n" + "=" * 80)
    
    # Step 1: Collect historical data
    print("\n[STEP 1/3] Collecting Historical Data...")
    print("-" * 80)
    
    # Import the collector module
    from nfl_data_collector import FantasyPlayerDataCollector
    
    collector = FantasyPlayerDataCollector()
    df_history = collector.combine_data(years=HISTORICAL_YEARS)
    
    print(f"\n✅ Historical data collected: {len(df_history)} records")
    print(f"   Saved to: nfl_player_data_with_history.csv")
    
    # Step 2: Condense data for predictions (exclude target season stats)
    print("\n[STEP 2/3] Condensing Data for Predictions...")
    print("-" * 80)
    
    # Import the condense module
    from condense_player_data import condense_player_data
    
    df_condensed = condense_player_data(
        input_csv='nfl_player_data_with_history.csv',
        output_csv='nfl_players_condensed.csv',
        target_season=TARGET_SEASON
    )
    
    print(f"\n✅ Condensed data created: {len(df_condensed)} players")
    print(f"   Saved to: nfl_players_condensed.csv")
    
    # Step 3: Extract target season actual stats
    print(f"\n[STEP 3/3] Extracting {TARGET_SEASON} Actual Stats...")
    print("-" * 80)
    
    # Import the extract module
    from extract_2024_stats import extract_2024_stats
    
    df_actual = extract_2024_stats(
        input_csv='nfl_player_data_with_history.csv',
        output_csv=f'nfl_players_{TARGET_SEASON}_stats.csv'
    )
    
    print(f"\n✅ {TARGET_SEASON} stats extracted: {len(df_actual)} players")
    print(f"   Saved to: nfl_players_{TARGET_SEASON}_stats.csv")
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print(f"  1. nfl_player_data_with_history.csv - All historical data ({min(HISTORICAL_YEARS)}-{max(HISTORICAL_YEARS)})")
    print(f"  2. nfl_players_condensed.csv - Weighted averages for predictions")
    print(f"     • Stats from: {min(HISTORICAL_YEARS)}-{TARGET_SEASON-1}")
    print(f"     • ADP from: {TARGET_SEASON}")
    print(f"     • {len(df_condensed)} players total")
    print(f"  3. nfl_players_{TARGET_SEASON}_stats.csv - Actual {TARGET_SEASON} performance")
    print(f"     • {len(df_actual)} players total")
    
    # Show rookies count
    rookies_in_condensed = df_condensed[df_condensed['is_rookie'] == True]
    print(f"\nRookies in prediction data: {len(rookies_in_condensed)} (stats all zeros, only ADP)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()