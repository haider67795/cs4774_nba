import pandas as pd


def normalize_nba_data(input_file, output_file, is_playoffs=False):
    try:
        df = pd.read_csv(input_file)
        print(f"Processing {input_file}...")

        # 1. NET_RATING: Point differential per 100 possessions
        # (This is the 'smarter' version of Plus/Minus)
        df['NET_RATING'] = (df['PLUS_MINUS'] / df['POSS']) * 100

        # 2. Rate-based stats (Per 100 Possessions)
        # Normalizes game-control stats across different season/series lengths
        stats_to_normalize = ['AST', 'REB', 'TOV', 'BLK']
        for stat in stats_to_normalize:
            df[f'{stat}_PER_100'] = (df[stat] / df['POSS']) * 100

        # 3. Save the new file
        df.to_csv(output_file, index=False)
        print(f"Success! Normalized data saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Check your file names!")
    except Exception as e:
        print(f"An error occurred: {e}")


# Apply to Regular Season
normalize_nba_data('nba_regular_season_2004_2026.csv',
                   'nba_reg_normalized.csv')

# Apply to Playoffs
normalize_nba_data('nba_playoffs_2004_2026.csv', 'nba_ply_normalized.csv')
