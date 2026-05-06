import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import teamdashlineups
from nba_api.stats.endpoints import synergyplaytypes
from nba_api.stats.endpoints import leaguedashplayerstats

def calculate_std_dev_pie_help(id,season_type,seas):
    print(id)
    time.sleep(2)
    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        team_id_nullable=id,
        measure_type_detailed_defense='Advanced', # Required for PIE
        season=seas,
        season_type_all_star=season_type
    )
    df = player_stats.get_data_frames()[0]
    print(df['PIE'].std())
    return df['PIE'].std()

def calculate_std_dev_pie(id,season_type,seas):
    print(id)
    id['std'] = 0.1
    for index, row in id.iterrows():
        id.at[index,'std'] = calculate_std_dev_pie_help(id.at[index,'TEAM_ID'],season_type,seas)
    print(id)
    return id['std']

def get_lineup_dataframe(id,season_type,seas):
    lineup_stats = teamdashlineups.TeamDashLineups(
        team_id=id,
        measure_type_detailed_defense='Advanced',
        group_quantity=4,
        season=seas,
        season_type_all_star=season_type
    )
    return lineup_stats.get_data_frames()[1]

def calculate_std_dev_playtime_top_ten_lineup(id,season_type,seas):
    time.sleep(0.1)
    lineup_stats = teamdashlineups.TeamDashLineups(
        team_id=id,
        measure_type_detailed_defense='Advanced',
        group_quantity=4,
        season=seas,
        season_type_all_star=season_type
    )
    df = lineup_stats.get_data_frames()[1]
    df.sort_values(by="SUM_TIME_PLAYED")
    top_ten_lineups_by_time_played = df[0:10]
    sum_top_ten_lineups_playtime = sum(df['SUM_TIME_PLAYED'])
    top_ten_lineups_by_time_played['PERCENT_TIME_PLAYED_WRT_TOP_TEN'] = top_ten_lineups_by_time_played['SUM_TIME_PLAYED']/sum_top_ten_lineups_playtime
    std_dev_lineups = top_ten_lineups_by_time_played['PERCENT_TIME_PLAYED_WRT_TOP_TEN'].std()
    return std_dev_lineups

def get_playtype_percent(ptype,season_type,seas):
    request = synergyplaytypes.SynergyPlayTypes(
        play_type_nullable=ptype,
        player_or_team_abbreviation='T',
        type_grouping_nullable='offensive',
        season=seas,
        season_type_all_star=season_type,
        per_mode_simple='PerGame'
    )
    df = request.get_data_frames()[0]
    df = df[['TEAM_ID','POSS_PCT']]
    newName = 'POSS_PCT_' + ptype
    df = df.rename(columns={'POSS_PCT':newName})
    return df

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

                # Calculate POSS_PCT_playtype
                #play_types = ['Isolation','Transition','PRBallHandler','PRRollMan','Postup','Spotup','Handoff','Cut','OffScreen','Misc']
                #for play in play_types:
                #    print(play)
                #    time.sleep(0.6)
                #    play_df = get_playtype_percent(play,season_type,season)
                #    raw_stats = pd.merge(raw_stats,play_df,on='TEAM_ID')

                #raw_stats['PIE_CON'] = calculate_std_dev_pie(raw_stats[['TEAM_ID']],season_type,season)

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
