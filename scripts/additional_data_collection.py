import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import teamdashlineups
from nba_api.stats.endpoints import synergyplaytypes

def get_lineup_dataframe(id,season_type,seas):
    print("WORD")
    lineup_stats = teamdashlineups.TeamDashLineups(
        team_id=id,
        measure_type_detailed_defense='Advanced',
        group_quantity=4,
        season=seas,
        season_type_all_star=season_type
    )
    print("WOW")
    print(lineup_stats.get_data_frames()[1])
    return lineup_stats.get_data_frames()[1]

def calculate_std_dev_playtime_top_ten_lineup(df):
    df.sort_values(by="SUM_TIME_PLAYED")
    top_ten_lineups_by_time_played = df[0:10]
    sum_top_ten_lineups_playtime = sum(top_ten_lineups_by_time_played['SUM_TIME_PLAYED'])
    top_ten_lineups_by_time_played['PERCENT_TIME_PLAYED_WRT_TOP_TEN'] = top_ten_lineups_by_time_played['SUM_TIME_PLAYED']/sum_top_ten_lineups_playtime
    std_dev_lineups = top_ten_lineups_by_time_played['PERCENT_TIME_PLAYED_WRT_TOP_TEN'].std()
    return std_dev_lineups

#df = get_lineup_dataframe('1610612764','Regular Season','2006-07')
#print(calculate_std_dev_playtime_top_ten_lineup(df))

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

#play_types = ['Isolation','Transition','PRBallHandler','PRRollMan','PostUp','SpotUp','HandOff','Cut','OffScreen','Putback','Misc']

#df = get_playtype_percent('Isolation','Playoffs','2012-13')
#print(df)

from nba_api.stats.endpoints import leaguedashplayerstats

def calculate_std_dev_pie(id,season_type,seas):
    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        team_id_nullable=id,
        measure_type_detailed_defense='Advanced', # Required for PIE
        season=seas,
        season_type_all_star=season_type
    )
    df = player_stats.get_data_frames()[0]
    return df['PIE'].std()

# Replace '1610612744' with your desired Team ID (e.g., Golden State Warriors)
team_id = '1610612744'

# Fetch advanced stats for the current season
player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
    team_id_nullable=team_id,
    measure_type_detailed_defense='Advanced', # Required for PIE
    season='2023-24'
)

print(calculate_std_dev_pie(team_id,"Regular Season",'2004-05'))
