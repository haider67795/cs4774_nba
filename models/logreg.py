import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
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

# --------------------------------------------------
# 9. EXTRA EDA: ROC, PR, AND CALIBRATION PLOTS
# --------------------------------------------------
from pathlib import Path

EDA_DIR = Path("eda_logreg")
EDA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Choose the most recent season with valid labels for BOTH tasks.
# This avoids errors when 2025-26 is still incomplete.
# --------------------------------------------------
eda_df = None
eda_conf_df = None
eda_season = None

for season in sorted(df["SEASON_ID"].dropna().unique(), reverse=True):
    season_df = df[df["SEASON_ID"] == season].copy()

    # Need both classes present for playoff qualification EDA
    if season_df["MADE_PLAYOFFS"].nunique() < 2:
        continue

    conf_df = season_df[season_df["MADE_PLAYOFFS"] == 1].copy()

    # Need both classes present for conference-finals EDA
    if len(conf_df) == 0 or conf_df["MADE_CONF_FINALS"].nunique() < 2:
        continue

    eda_df = season_df
    eda_conf_df = conf_df
    eda_season = season
    break

if eda_df is None:
    print("WARNING: No completed season found with valid labels for extra EDA. Skipping Section 9.")
else:
    print(f"\nUsing {eda_season} for extra EDA plots.\n")

    # Predict probabilities for the selected EDA season
    eda_df["PROB_QUAL"] = model_qual.predict_proba(
        scaler_qual.transform(eda_df[features])
    )[:, 1]

    eda_conf_df["PROB_CONF"] = model_conf.predict_proba(
        scaler_conf.transform(eda_conf_df[features])
    )[:, 1]

    # ------------------------------
    # Plot 1: ROC Curves
    # ------------------------------
    plt.figure(figsize=(9, 7))

    fpr_q, tpr_q, _ = roc_curve(eda_df["MADE_PLAYOFFS"], eda_df["PROB_QUAL"])
    auc_q = auc(fpr_q, tpr_q)
    plt.plot(fpr_q, tpr_q, linewidth=2, label=f"Playoff Qualification (AUC = {auc_q:.3f})")

    fpr_c, tpr_c, _ = roc_curve(eda_conf_df["MADE_CONF_FINALS"], eda_conf_df["PROB_CONF"])
    auc_c = auc(fpr_c, tpr_c)
    plt.plot(fpr_c, tpr_c, linewidth=2, label=f"Conference Finals (AUC = {auc_c:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: Qualification vs. Conference Finals ({eda_season})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "roc_curves.png", dpi=200)
    plt.close()

    # ------------------------------
    # Plot 2: Precision-Recall Curves
    # ------------------------------
    plt.figure(figsize=(9, 7))

    precision_q, recall_q, _ = precision_recall_curve(
        eda_df["MADE_PLAYOFFS"], eda_df["PROB_QUAL"]
    )
    ap_q = average_precision_score(eda_df["MADE_PLAYOFFS"], eda_df["PROB_QUAL"])
    plt.plot(recall_q, precision_q, linewidth=2,
             label=f"Playoff Qualification (AP = {ap_q:.3f})")

    precision_c, recall_c, _ = precision_recall_curve(
        eda_conf_df["MADE_CONF_FINALS"], eda_conf_df["PROB_CONF"]
    )
    ap_c = average_precision_score(eda_conf_df["MADE_CONF_FINALS"], eda_conf_df["PROB_CONF"])
    plt.plot(recall_c, precision_c, linewidth=2,
             label=f"Conference Finals (AP = {ap_c:.3f})")

    plt.hlines(
        eda_df["MADE_PLAYOFFS"].mean(),
        xmin=0,
        xmax=1,
        linestyles="--",
        linewidth=1,
        label=f"Qualification baseline = {eda_df['MADE_PLAYOFFS'].mean():.3f}"
    )
    plt.hlines(
        eda_conf_df["MADE_CONF_FINALS"].mean(),
        xmin=0,
        xmax=1,
        linestyles=":",
        linewidth=1,
        label=f"Conference Finals baseline = {eda_conf_df['MADE_CONF_FINALS'].mean():.3f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves: Qualification vs. Conference Finals ({eda_season})")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "precision_recall_curves.png", dpi=200)
    plt.close()

    # ------------------------------
    # Plot 3: Calibration Curves
    # ------------------------------
    plt.figure(figsize=(9, 7))

    frac_pos_q, mean_pred_q = calibration_curve(
        eda_df["MADE_PLAYOFFS"], eda_df["PROB_QUAL"], n_bins=8, strategy="quantile"
    )
    brier_q = brier_score_loss(eda_df["MADE_PLAYOFFS"], eda_df["PROB_QUAL"])
    plt.plot(mean_pred_q, frac_pos_q, marker="o", linewidth=2,
             label=f"Playoff Qualification (Brier = {brier_q:.3f})")

    frac_pos_c, mean_pred_c = calibration_curve(
        eda_conf_df["MADE_CONF_FINALS"], eda_conf_df["PROB_CONF"], n_bins=8, strategy="quantile"
    )
    brier_c = brier_score_loss(eda_conf_df["MADE_CONF_FINALS"], eda_conf_df["PROB_CONF"])
    plt.plot(mean_pred_c, frac_pos_c, marker="o", linewidth=2,
             label=f"Conference Finals (Brier = {brier_c:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction Positive")
    plt.title(f"Calibration Curves: Qualification vs. Conference Finals ({eda_season})")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "calibration_curves.png", dpi=200)
    plt.close()

    # ------------------------------
    # Summary metrics
    # ------------------------------
    print("\n=== Extra Logistic Regression EDA ===\n")
    print(f"Season used for EDA: {eda_season}\n")

    print(f"Playoff Qualification ROC AUC: {auc_q:.4f}")
    print(f"Playoff Qualification AP:      {ap_q:.4f}")
    print(f"Playoff Qualification Brier:   {brier_q:.4f}\n")

    print(f"Conference Finals ROC AUC:     {auc_c:.4f}")
    print(f"Conference Finals AP:          {ap_c:.4f}")
    print(f"Conference Finals Brier:       {brier_c:.4f}\n")

    print(f"Extra EDA plots saved to: {EDA_DIR.resolve()}")

    # ROC AUC interpretation: 
    # Playoff Qualification model has high Auc (~0.9)
    # model is very good at distinguishing playoff vs non-playoff teams,
    #makes sense because Net Rating dominates
    #Conference Finals model: has lower AUC ( ~0.7):
    #harder problem: We're distinguishing elite teams from other elite teams
    #reflects real NBA uncertainty

    