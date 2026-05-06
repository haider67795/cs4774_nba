import pandas as pd

# 1. Load your existing data
# Update these paths to match your local file names
REG_SEASON_PATH = "data/nba_reg_normalized.csv"
PLAYOFF_DATA_PATH = "data/nba_playoffs_2004_2026.csv"
OUTPUT_PATH = "data/nba_labeled_dataset.csv"

reg_df = pd.read_csv(REG_SEASON_PATH)
ply_df = pd.read_csv(PLAYOFF_DATA_PATH)

print(
    f"Loaded {len(reg_df)} regular season rows and {len(ply_df)} playoff rows.")

# --------------------------------------------------
# 2. Target Engineering (Ground Truth)
# --------------------------------------------------

# Identify teams that qualified for the playoffs
playoff_keys = ply_df[["TEAM_ID", "SEASON_ID"]].drop_duplicates().copy()
playoff_keys["MADE_PLAYOFFS"] = 1

# Merge qualification status onto regular season data
df = pd.merge(reg_df, playoff_keys, on=["TEAM_ID", "SEASON_ID"], how="left")
df["MADE_PLAYOFFS"] = df["MADE_PLAYOFFS"].fillna(0).astype(int)

# Map playoff win counts to determine advancement
wins_map = ply_df[["TEAM_ID", "SEASON_ID", "W"]].copy()
wins_map = wins_map.rename(columns={"W": "PLAYOFF_WINS"})

# Merge wins and create the Conference Finals target (8+ wins)
df = pd.merge(df, wins_map, on=["TEAM_ID", "SEASON_ID"], how="left")
df["PLAYOFF_WINS"] = df["PLAYOFF_WINS"].fillna(0)
df["MADE_CONF_FINALS"] = (df["PLAYOFF_WINS"] >= 8).astype(int)

# --------------------------------------------------
# 3. Export Labeled Dataset
# --------------------------------------------------

# Clean up: You might want to drop PLAYOFF_WINS before saving if it's only for the label
# df = df.drop(columns=["PLAYOFF_WINS"])

df.to_csv(OUTPUT_PATH, index=False)

print(f"Successfully created labeled dataset at: {OUTPUT_PATH}")
print(f"Total Playoff Qualifiers: {df['MADE_PLAYOFFS'].sum()}")
print(f"Total Conf. Finals Teams: {df['MADE_CONF_FINALS'].sum()}")
