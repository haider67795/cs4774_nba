import pandas as pd
import numpy as np


def validate_nba_dataset(file_path, expected_rows, dataset_name):
    print(f"=== Validating {dataset_name} ===")
    df = pd.read_csv(file_path)
    errors = 0

    # 1. Check Row Counts
    actual_rows = len(df)
    if actual_rows == expected_rows:
        print(f"✅ Row Count: Matches ({actual_rows})")
    else:
        print(
            f"❌ Row Count: Mismatch! Found {actual_rows}, expected {expected_rows}")
        errors += 1

    # 2. Check for Missing Values (NaNs)
    null_counts = df.isnull().sum().sum()
    if null_counts == 0:
        print(f"✅ Null Values: None found.")
    else:
        print(f"❌ Null Values: Found {null_counts} missing values!")
        errors += 1

    # 3. Check for Duplicates (Team + Season + Type)
    duplicates = df.duplicated(
        subset=['TEAM_ID', 'SEASON_ID', 'SEASON_TYPE']).sum()
    if duplicates == 0:
        print(f"✅ Duplicates: No redundant team entries.")
    else:
        print(
            f"❌ Duplicates: Found {duplicates} duplicate team/season entries!")
        errors += 1

    # 4. Basketball Logic Test: W + L == GP
    logic_fail = df[df['W'] + df['L'] != df['GP']]
    if len(logic_fail) == 0:
        print(f"✅ Game Logic: Wins + Losses = Games Played for all rows.")
    else:
        print(
            f"❌ Game Logic: Found {len(logic_fail)} rows where W+L does not equal GP!")
        errors += 1

    # 5. Sanity Check: Custom Feature Ranges
    # Offensive Rating usually stays between 80 and 130
    outlier_rating = df[(df['OFF_RATING_CUSTOM'] < 80) |
                        (df['OFF_RATING_CUSTOM'] > 140)]
    if len(outlier_rating) == 0:
        print(f"✅ Efficiency Check: OFF_RATING_CUSTOM values are within realistic ranges.")
    else:
        print(
            f"⚠️ Efficiency Check: Found {len(outlier_rating)} potential outliers in Offensive Rating.")

    # 6. Check Percentage Ranges (0.0 to 1.0)
    pct_cols = ['W_PCT', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    pct_error = False
    for col in pct_cols:
        if df[col].max() > 1.0 or df[col].min() < 0.0:
            print(f"❌ Range Error: {col} has values outside 0.0-1.0!")
            pct_error = True
            errors += 1
    if not pct_error:
        print(f"✅ Percentage Check: All rate stats are normalized between 0 and 1.")

    print(f"\nConclusion: {dataset_name} has {errors} critical errors.\n")


# Run the validation
validate_nba_dataset('nba_regular_season_2004_2026.csv', 660, "Regular Season")
validate_nba_dataset('nba_playoffs_2004_2026.csv', 336, "Playoffs")
