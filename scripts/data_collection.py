import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashteamstats


def fetch_master_nba_data(start_year=2004, end_year=2025):
    all_regular_season = []
    all_playoffs = []

    # Generate season strings like '2004-05', '2005-06', etc.
    seasons = [f"{y}-{str(y+1)[2:]}" for y in range(start_year, end_year + 1)]

    for season in seasons:
        print(f"Fetching data for {season}...")

        for season_type in ['Regular Season', 'Playoffs']:
            try:
                # Fetch raw 'Base' stats
                raw_stats = leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    season_type_all_star=season_type,
                    measure_type_detailed_defense='Base'
                ).get_data_frames()[0]

                if raw_stats.empty:
                    continue

                # Add metadata
                raw_stats['SEASON_ID'] = season
                raw_stats['SEASON_TYPE'] = season_type

                # Calculate Possessions: 0.96 * (FGA + TOV + 0.44 * FTA - ORB)
                raw_stats['POSS'] = 0.96 * (
                    raw_stats['FGA'] +
                    raw_stats['TOV'] +
                    (0.44 * raw_stats['FTA']) -
                    raw_stats['OREB']
                )

                # Calculate Offensive Rating: (Points / Possessions) * 100
                raw_stats['OFF_RATING_CUSTOM'] = (
                    raw_stats['PTS'] / raw_stats['POSS']) * 100

                # Calculate Defensive Rating (using opponent stats is more complex,
                # but many use the API's built-in 'Advanced' endpoint for this).

                if season_type == 'Regular Season':
                    all_regular_season.append(raw_stats)
                else:
                    all_playoffs.append(raw_stats)

                # Sleep briefly to avoid hitting API rate limits
                time.sleep(0.6)

            except Exception as e:
                print(f"Error fetching {season_type} for {season}: {e}")

    # Combine into master DataFrames
    df_reg = pd.concat(all_regular_season, ignore_index=True)
    df_ply = pd.concat(all_playoffs, ignore_index=True)

    return df_reg, df_ply


# Run the collection
regular_season_df, playoffs_df = fetch_master_nba_data()

# Save for your EDA and XGBoost model
regular_season_df.to_csv('nba_regular_season_2004_2026.csv', index=False)
playoffs_df.to_csv('nba_playoffs_2004_2026.csv', index=False)

print("Data collection complete. Master CSVs saved.")
