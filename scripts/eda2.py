import pandas as pd
import numpy as np


def run_full_feature_audit(file_path='nba_model_training_data.csv'):
    df = pd.read_csv(file_path)

    # Filter for numerical columns and exclude metadata
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['TEAM_ID', 'PLAYOFF_WINS', 'MADE_PLAYOFFS', 'MADE_CONF_FINALS']
    features = [c for c in numerical_cols if c not in exclude]

    # Calculate correlation with the Conference Finals target
    # (Exclude 2025-26 since we don't have results for it yet)
    analysis_df = df[df['SEASON_ID'] != '2025-26']
    correlations = analysis_df[features + ['MADE_CONF_FINALS']
                               ].corr()['MADE_CONF_FINALS'].sort_values(ascending=False)

    print("--- Full Feature Audit: Correlation with Conference Finals ---")
    print(correlations)

    # Identify pairs to drop (Multicollinearity > 0.90)
    corr_matrix = analysis_df[features].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    print("\n--- Suggested Columns to Drop (Redundant) ---")
    print(to_drop)


if __name__ == "__main__":
    run_full_feature_audit()
