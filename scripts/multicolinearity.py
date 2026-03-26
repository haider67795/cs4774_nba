import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def create_nba_heatmaps():
    # Load files
    df_reg = pd.read_csv('nba_regular_season_2004_2026.csv')
    df_ply = pd.read_csv('nba_playoffs_2004_2026.csv')

    # Define a plotting helper
    def plot_matrix(df, title, filename):
        # Select only numerical columns and drop non-informative IDs
        num_df = df.select_dtypes(include=[np.number]).drop(
            columns=['TEAM_ID'], errors='ignore')

        plt.figure(figsize=(20, 16))
        corr = num_df.corr()

        # We use a diverging color map (RdBu_r) to see pos/neg correlations clearly
        sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=.5)
        plt.title(title, fontsize=22)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved: {filename}")

    # Generate both
    plot_matrix(df_reg, 'Correlation Matrix: All Regular Season Features',
                'reg_all_features.png')
    plot_matrix(df_ply, 'Correlation Matrix: All Playoff Features',
                'ply_all_features.png')


if __name__ == "__main__":
    create_nba_heatmaps()
