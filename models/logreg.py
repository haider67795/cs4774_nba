import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# --------------------------------------------------
# 1. DATA LOADING (2004 - 2026 Season Range) [cite: 6, 56]
# --------------------------------------------------
# Ensure these files are in your local directory
try:
    reg_df = pd.read_csv("data/nba_reg_normalized.csv")
    ply_df = pd.read_csv("data/nba_playoffs_2004_2026.csv")
except FileNotFoundError as e:
    print(f"Error: Missing required CSV files. {e}")
    exit()

# --------------------------------------------------
# 2. TARGET ENGINEERING (GROUND TRUTH) [cite: 9, 105]
# --------------------------------------------------

# Target 1: Playoff Qualification (Binary 0/1) [cite: 27]
playoff_keys = ply_df[["TEAM_ID", "SEASON_ID"]].drop_duplicates().copy()
playoff_keys["MADE_PLAYOFFS"] = 1
df = pd.merge(reg_df, playoff_keys, on=["TEAM_ID", "SEASON_ID"], how="left")
df["MADE_PLAYOFFS"] = df["MADE_PLAYOFFS"].fillna(0).astype(int)

# Target 2: Conference Finals Advancement (8+ Playoff Wins) [cite: 19, 74]
wins_map = ply_df[["TEAM_ID", "SEASON_ID", "W"]].copy()
wins_map = wins_map.rename(columns={"W": "PLAYOFF_WINS"})
df = pd.merge(df, wins_map, on=["TEAM_ID", "SEASON_ID"], how="left")
df["PLAYOFF_WINS"] = df["PLAYOFF_WINS"].fillna(0)
df["MADE_CONF_FINALS"] = (df["PLAYOFF_WINS"] >= 8).astype(int)

# --------------------------------------------------
# 3. FEATURE SELECTION (9 PACE-ADJUSTED METRICS) [cite: 12, 13]
# --------------------------------------------------
# features = [
#     "NET_RATING", "OFF_RATING_CUSTOM", "FG_PCT", "FG3_PCT",
#     "FT_PCT", "AST_PER_100", "REB_PER_100", "TOV_PER_100", "BLK_PER_100"
# ]

# features = ["OFF_RATING_CUSTOM", "FG_PCT", "FG3_PCT",
#     "FT_PCT", "AST_PER_100", "REB_PER_100", "TOV_PER_100", "BLK_PER_100"
# ]

features = [
    "OFF_RATING_CUSTOM",
    "DEF_RATING_CUSTOM",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "AST_PER_100",
    "REB_PER_100",
    "TOV_PER_100"
]
df = df.dropna(subset=features).copy()

# --------------------------------------------------
# 4. TEMPORAL TRAIN/TEST SPLIT [cite: 97]
# --------------------------------------------------
train_df = df[df["SEASON_ID"] != "2025-26"].copy()
test_df = df[df["SEASON_ID"] == "2025-26"].copy()

# --------------------------------------------------
# 5. MODEL A: PLAYOFF QUALIFICATION [cite: 27, 82]
# --------------------------------------------------
X_train_qual = train_df[features]
y_train_qual = train_df["MADE_PLAYOFFS"]

scaler_qual = StandardScaler()
X_train_qual_scaled = scaler_qual.fit_transform(X_train_qual)

model_qual = LogisticRegression(max_iter=1000, random_state=42)
model_qual.fit(X_train_qual_scaled, y_train_qual)

# --------------------------------------------------
# 6. MODEL B: CONFERENCE FINALS ADVANCEMENT [cite: 27, 74]
# --------------------------------------------------
train_conf_df = train_df[train_df["MADE_PLAYOFFS"] == 1].copy()
X_train_conf = train_conf_df[features]
y_train_conf = train_conf_df["MADE_CONF_FINALS"]

scaler_conf = StandardScaler()
X_train_conf_scaled = scaler_conf.fit_transform(X_train_conf)

model_conf = LogisticRegression(max_iter=1000, random_state=42)
model_conf.fit(X_train_conf_scaled, y_train_conf)

# --------------------------------------------------
# 7. 2026 PREDICTIONS (OPTION B: PREDICTIVE CHAIN)
# --------------------------------------------------
X_test_all = test_df[features]
X_test_qual_scaled = scaler_qual.transform(X_test_all)

# Step 1: Predict Playoff Entry probability
test_df["PROB_QUAL"] = model_qual.predict_proba(X_test_qual_scaled)[:, 1]

# Step 2: Apply Model B to predicted qualifiers (Threshold >= 0.5)
test_df["PROB_CONF"] = 0.0
predicted_mask = test_df["PROB_QUAL"] >= 0.5
if predicted_mask.any():
    X_pred_conf = test_df.loc[predicted_mask, features]
    X_pred_conf_scaled = scaler_conf.transform(X_pred_conf)
    test_df.loc[predicted_mask, "PROB_CONF"] = model_conf.predict_proba(
        X_pred_conf_scaled)[:, 1]

# Save Results
test_df[["TEAM_NAME", "PROB_QUAL", "PROB_CONF"]].sort_values(
    by="PROB_CONF", ascending=False).to_csv("nba_2026_predictions.csv", index=False)

# --------------------------------------------------
# 8. VISUALIZATION: COEFFICIENT SHIFT [cite: 16, 81]
# --------------------------------------------------


def generate_graphs(model_qual, model_conf, features):
    # Prepare Dataframe for Coefficients
    coef_df = pd.DataFrame({
        'Feature': features,
        'Qualification': model_qual.coef_[0],
        'Conference Finals': model_conf.coef_[0]
    }).melt(id_vars='Feature', var_name='Model', value_name='Weight')

    # Graph 1: Coefficient Comparison (The "Shift")
    plt.figure(figsize=(12, 7))
    sns.barplot(data=coef_df, x='Weight', y='Feature',
                hue='Model', palette='viridis')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title(
        'Predictive Weight Shift: Regular Season vs. Postseason [cite: 16, 81]')
    plt.xlabel('Logistic Regression Coefficient (Standardized)')
    plt.tight_layout()
    plt.savefig('coefficient_shift.png')

    # Graph 2: Training Metrics (F1 and Accuracy) [cite: 97]
    train_metrics = {
        'Model': ['Qualification', 'Conf. Finals'],
        'Accuracy': [
            accuracy_score(y_train_qual, model_qual.predict(
                X_train_qual_scaled)),
            accuracy_score(
                y_train_conf, model_conf.predict(X_train_conf_scaled))
        ],
        'F1-Score': [
            f1_score(y_train_qual, model_qual.predict(X_train_qual_scaled)),
            f1_score(y_train_conf, model_conf.predict(X_train_conf_scaled))
        ]
    }

    perf_df = pd.DataFrame(train_metrics).melt(
        id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=perf_df, x='Model', y='Score',
                hue='Metric', palette='magma')
    plt.title('Training Performance Evaluation [cite: 95, 97]')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('model_performance.png')


generate_graphs(model_qual, model_conf, features)
print("\nSuccess! Models trained, predictions saved, and graphs generated.")
