import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def prepare_and_analyze_nba_data(reg_path, ply_path, output_path):
    # 1. Load Datasets
    reg = pd.read_csv(reg_path)
    ply = pd.read_csv(ply_path)

    # 2. Extract Playoff Outcomes
    # We need the number of wins and a flag for participation
    ply_stats = ply[['TEAM_ID', 'SEASON_ID', 'W']].rename(
        columns={'W': 'PLAYOFF_WINS'})
    ply_participants = ply[['TEAM_ID', 'SEASON_ID']].copy()
    ply_participants['MADE_PLAYOFFS'] = 1

    # 3. Merge Regular Season Data with Playoff Labels
    # We use a Left Join so teams that missed the playoffs stay in the dataset with 0 wins
    combined = pd.merge(reg, ply_stats, on=[
                        'TEAM_ID', 'SEASON_ID'], how='left')
    combined = pd.merge(combined, ply_participants, on=[
                        'TEAM_ID', 'SEASON_ID'], how='left')

    # 4. Handle Missing Values for Non-Playoff Teams
    combined['PLAYOFF_WINS'] = combined['PLAYOFF_WINS'].fillna(0).astype(int)
    combined['MADE_PLAYOFFS'] = combined['MADE_PLAYOFFS'].fillna(0).astype(int)

    # Define Target: Conference Finals (Winning at least 2 rounds = 8 wins)
    combined['MADE_CONF_FINALS'] = (combined['PLAYOFF_WINS'] >= 8).astype(int)

    # 5. Automated EDA (Exploratory Data Analysis)
    # We exclude the current 2025-26 season for analysis because the outcomes are unknown
    analysis_df = combined[combined['SEASON_ID'] != '2025-26'].copy()

    # Feature Selection for Correlation
    features = [
        'W_PCT', 'PLUS_MINUS', 'OFF_RATING_CUSTOM', 'FG_PCT',
        'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV', 'STL', 'BLK'
    ]
    targets = ['MADE_PLAYOFFS', 'MADE_CONF_FINALS']

    # A. Pairwise Correlation (Multicollinearity Check)
    corr_matrix = analysis_df[features].corr()

    print("--- Multicollinearity Check ---")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.85:
                colname = corr_matrix.columns[i]
                refname = corr_matrix.columns[j]
                high_corr.append(
                    f"{colname} vs {refname}: {corr_matrix.iloc[i, j]:.2f}")

    if high_corr:
        print("Warning: High correlation detected between features (potential redundancy):")
        for line in high_corr:
            print(f"  - {line}")
    else:
        print("No extreme multicollinearity detected.")

    # B. Relationship with Success Targets
    print("\n--- Correlation with Postseason Success ---")
    target_corr = analysis_df[features + targets].corr()[targets].drop(targets)
    print(target_corr.sort_values(by='MADE_CONF_FINALS', ascending=False))

    # C. Statistical Profile Comparison
    print("\n--- Feature Averages by Conference Finals Achievement ---")
    profiles = analysis_df.groupby('MADE_CONF_FINALS')[features].mean()
    print(profiles)

    # 6. Save Processed Dataset for XGBoost Training
    combined.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved analysis-ready dataset to: {output_path}")

    # 7. Visualization: Feature Correlation Heatmap
    plt.figure(figsize=(12, 10))
    full_corr = analysis_df[features + targets].corr()
    sns.heatmap(full_corr, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
    plt.title('NBA Feature Correlation: Regular Season vs Playoff Success')
    plt.tight_layout()
    plt.savefig('nba_feature_heatmap.png')
    print("Correlation heatmap saved as 'nba_feature_heatmap.png'")


if __name__ == "__main__":
    prepare_and_analyze_nba_data(
        reg_path='nba_regular_season_2004_2026.csv',
        ply_path='nba_playoffs_2004_2026.csv',
        output_path='nba_model_training_data.csv'
    )
