import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamelog


def final_robust_patch(reg_file='data/nba_regular_season_2004_2026.csv', ply_file='data/nba_playoffs_2004_2026.csv'):
    # Targeting the specific failures from your last run
    targets = [
        ('2011-12', 'Playoffs'),
        ('2022-23', 'Playoffs'),
        ('2023-24', 'Playoffs'),
        ('2024-25', 'Playoffs')
    ]

    for season, s_type in targets:
        print(f"--- Fixing {season} {s_type} (Zero-Win Handling) ---")
        try:
            # 1. Fetch raw logs
            log = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star=s_type).get_data_frames()[0]

            # 2. Aggregate raw totals
            agg_map = {
                'MIN': 'sum', 'FGM': 'sum', 'FGA': 'sum', 'FG3M': 'sum', 'FG3A': 'sum',
                'FTM': 'sum', 'FTA': 'sum', 'OREB': 'sum', 'DREB': 'sum', 'REB': 'sum',
                'AST': 'sum', 'STL': 'sum', 'BLK': 'sum', 'TOV': 'sum', 'PF': 'sum',
                'PTS': 'sum', 'PLUS_MINUS': 'sum'
            }
            df_agg = log.groupby(['TEAM_ID', 'TEAM_NAME']
                                 ).agg(agg_map).reset_index()

            # 3. ROBUST WIN/LOSS COUNTING (Fixes the Length Mismatch)
            # This counts every W and L, and fills missing ones (sweeps) with 0
            wl_counts = log.groupby('TEAM_ID')[
                'WL'].value_counts().unstack(fill_value=0)

            # Ensure 'W' and 'L' columns exist even if no one won/lost (rare but safe)
            if 'W' not in wl_counts:
                wl_counts['W'] = 0
            if 'L' not in wl_counts:
                wl_counts['L'] = 0

            # Merge wins/losses back into the main table
            df_agg = df_agg.merge(wl_counts[['W', 'L']], on='TEAM_ID')
            df_agg['GP'] = df_agg['W'] + df_agg['L']
            df_agg['W_PCT'] = df_agg['W'] / df_agg['GP']

            # 4. Fill missing technical columns to match your CSV structure
            df_agg['FG_PCT'] = df_agg['FGM'] / df_agg['FGA']
            df_agg['FG3_PCT'] = df_agg['FG3M'] / df_agg['FG3A']
            df_agg['FT_PCT'] = df_agg['FTM'] / df_agg['FTA']
            df_agg['BLKA'] = 0  # Not in GameLog, set to 0 for structure
            df_agg['PFD'] = 0  # Not in GameLog, set to 0 for structure

            # 5. Metadata & Custom Features
            df_agg['SEASON_ID'] = season
            df_agg['SEASON_TYPE'] = s_type
            df_agg['POSS'] = 0.96 * (df_agg['FGA'] + df_agg['TOV'] +
                                     (0.44 * df_agg['FTA']) - df_agg['OREB'])
            df_agg['OFF_RATING_CUSTOM'] = (
                df_agg['PTS'] / df_agg['POSS']) * 100



            df_agg['OPP_PTS'] = df_agg['PTS'] - df_agg['PLUS_MINUS']
            df_agg['DEF_RATING_CUSTOM'] = (df_agg['OPP_PTS'] / df_agg['POSS']) * 100

            # 6. Re-calculate League Ranks for this season
            rank_cols = ['GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M',
                         'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
                         'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']
            for col in rank_cols:
                # Lower is better for these
                asc = col in ['TOV', 'PF', 'L', 'BLKA']
                df_agg[f'{col}_RANK'] = df_agg[col].rank(
                    ascending=asc, method='min').astype(int)

            # 7. Merge and Save
            target = reg_file if s_type == 'Regular Season' else ply_file
            master = pd.read_csv(target)
            master = master[~((master['SEASON_ID'] == season)
                              & (master['SEASON_TYPE'] == s_type))]

            # Align columns exactly to your CSV
            df_agg = df_agg[master.columns]
            pd.concat([master, df_agg], ignore_index=True).sort_values(
                'SEASON_ID').to_csv(target, index=False)

            print(f"SUCCESS: {season} {s_type} fully patched.")
            time.sleep(2)

        except Exception as e:
            print(f"FAILED {season} {s_type}: {e}")



import pandas as pd

def add_defensive_metrics(reg_file='data/nba_regular_season_2004_2026.csv', ply_file='data/nba_playoffs_2004_2026.csv'):
    # Process both the regular season and playoff files
    files = [reg_file, ply_file]

    for file_path in files:
        print(f"Processing {file_path}...")
        try:
            # 1. Load the existing data
            df = pd.read_csv(file_path)

            # 2. Ensure POSS is calculated (re-calculating just to be safe)
            if 'POSS' not in df.columns:
                df['POSS'] = 0.96 * (df['FGA'] + df['TOV'] + (0.44 * df['FTA']) - df['OREB'])

            # 3. Calculate Opponent Points
            df['OPP_PTS'] = df['PTS'] - df['PLUS_MINUS']

            # 4. Calculate Defensive Rating (Points Allowed per 100 Possessions)
            df['DEF_RATING_CUSTOM'] = (df['OPP_PTS'] / df['POSS']) * 100

            # 5. Calculate Stocks (Steals + Blocks) per 100 Possessions
            df['STOCKS_PER_100'] = ((df['STL'] + df['BLK']) / df['POSS']) * 100

            # 6. Overwrite the original CSV with the new columns included
            df.to_csv(file_path, index=False)
            
            print(f"SUCCESS: Defensive metrics added to {file_path}.")

        except Exception as e:
            print(f"FAILED on {file_path}: {e}")

if __name__ == "__main__":
    add_defensive_metrics()

