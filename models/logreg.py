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
from pathlib import Path

# --------------------------------------------------
# 1. DATA LOADING (2004 - 2026 Season Range)
# --------------------------------------------------
try:
    df = pd.read_csv("data/(main)nba_labeled_dataset.csv")
except FileNotFoundError as e:
    print(f"Error: Missing required CSV files. {e}")
    exit()

# --------------------------------------------------
# 3. FEATURE SELECTION (8 PACE-ADJUSTED METRICS)
# --------------------------------------------------
features = [
    "OFF_RATING_CUSTOM",
    "DEF_RATING_CUSTOM",
    # "FG_PCT",
    # "FG3_PCT",
    "FT_PCT",
    "AST_PER_100",
    "REB_PER_100",
    "TOV_PER_100"
]
df = df.dropna(subset=features).copy()

# --------------------------------------------------
# 4. TEMPORAL TRAIN/TEST SPLIT (FIXED 80/20)
# --------------------------------------------------
live_df = df[df["SEASON_ID"] == "2025-26"].copy()
history_df = df[df["SEASON_ID"] != "2025-26"].copy()

seasons = sorted(history_df["SEASON_ID"].unique())
cutoff_idx = int(len(seasons) * 0.8)
train_seasons = seasons[:cutoff_idx]
eval_seasons = seasons[cutoff_idx:]

train_df = history_df[history_df["SEASON_ID"].isin(train_seasons)].copy()
eval_df = history_df[history_df["SEASON_ID"].isin(eval_seasons)].copy()

print(f"Total historical seasons: {len(seasons)}")
print(
    f"Training on: {train_seasons[0]} to {train_seasons[-1]} ({len(train_seasons)} seasons)")
print(
    f"Evaluating on: {eval_seasons[0]} to {eval_seasons[-1]} ({len(eval_seasons)} seasons)")

# --------------------------------------------------
# 5. MODEL A: PLAYOFF QUALIFICATION
# --------------------------------------------------
X_train_qual = train_df[features]
y_train_qual = train_df["MADE_PLAYOFFS"]

scaler_qual = StandardScaler()
X_train_qual_scaled = scaler_qual.fit_transform(X_train_qual)

model_qual = LogisticRegression(max_iter=1000, random_state=42)
model_qual.fit(X_train_qual_scaled, y_train_qual)

# --------------------------------------------------
# 6. MODEL B: CONFERENCE FINALS ADVANCEMENT
# --------------------------------------------------
train_conf_df = train_df[train_df["MADE_PLAYOFFS"] == 1].copy()
X_train_conf = train_conf_df[features]
y_train_conf = train_conf_df["MADE_CONF_FINALS"]

scaler_conf = StandardScaler()
X_train_conf_scaled = scaler_conf.fit_transform(X_train_conf)

model_conf = LogisticRegression(max_iter=1000, random_state=42)
model_conf.fit(X_train_conf_scaled, y_train_conf)

# --------------------------------------------------
# NEW: PRINT MODEL COEFFICIENTS
# --------------------------------------------------
print("\n" + "="*50)
print("LOGISTIC REGRESSION COEFFICIENTS (Standardized)")
print("="*50)

# Create a DataFrame for easy reading
coef_summary = pd.DataFrame({
    'Feature': features,
    'Qual_Weight': model_qual.coef_[0],
    'Conf_Finals_Weight': model_conf.coef_[0]
})

print(coef_summary.to_string(index=False, justify='left'))
print("-" * 50)
print(f"Model Qual Intercept: {model_qual.intercept_[0]:.4f}")
print(f"Model Conf Intercept: {model_conf.intercept_[0]:.4f}")
print("="*50 + "\n")

# --------------------------------------------------
# 7. 2026 PREDICTIONS (TOP 16 & TOP 4 LOGIC)
# --------------------------------------------------
if not live_df.empty:
    X_live_scaled = scaler_qual.transform(live_df[features])
    live_df["PROB_QUAL"] = model_qual.predict_proba(X_live_scaled)[:, 1]
    live_df = live_df.sort_values(by="PROB_QUAL", ascending=False)
    live_df["PRED_MADE_PLAYOFFS"] = 0
    live_df.iloc[:16, live_df.columns.get_loc("PRED_MADE_PLAYOFFS")] = 1

    X_live_conf_scaled = scaler_conf.transform(live_df[features])
    live_df["PROB_CONF"] = model_conf.predict_proba(X_live_conf_scaled)[:, 1]
    live_df = live_df.sort_values(by="PROB_CONF", ascending=False)
    live_df["PRED_MADE_CONF_FINALS"] = 0
    live_df.iloc[:4, live_df.columns.get_loc("PRED_MADE_CONF_FINALS")] = 1

    output_cols = ["TEAM_NAME", "PROB_QUAL",
                   "PRED_MADE_PLAYOFFS", "PROB_CONF", "PRED_MADE_CONF_FINALS"]
    live_df[output_cols].to_csv(
        "./eval/logreg/nba_2026_predictions.csv", index=False)
    print("2026 Predictions saved to nba_2026_predictions.csv")

# --------------------------------------------------
# UPDATED PERFORMANCE EVALUATION
# --------------------------------------------------


def generate_coefficient_plot(model_qual, model_conf, features):
    coef_df = pd.DataFrame({
        'Feature': features,
        'Qualification': model_qual.coef_[0],
        'Conference Finals': model_conf.coef_[0]
    }).melt(id_vars='Feature', var_name='Model', value_name='Weight')

    plt.figure(figsize=(12, 7))
    sns.barplot(data=coef_df, x='Weight', y='Feature',
                hue='Model', palette='viridis')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Predictive Weight Shift: Making Playoffs vs. Deep Run')
    plt.xlabel('Standardized Coefficient (Importance)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./eval/logreg/coefficient_shift.png')
    print("Graph saved: coefficient_shift.png")


def generate_detailed_eval(model_qual, model_conf, features,
                           X_train_qual, y_train_qual, X_eval_qual, y_eval_qual,
                           X_train_conf, y_train_conf, X_eval_conf, y_eval_conf):
    def get_metrics(model, X, y, label, task):
        preds = model.predict(X)
        return [
            {'Dataset': label, 'Task': task, 'Metric': 'Accuracy',
                'Score': accuracy_score(y, preds)},
            {'Dataset': label, 'Task': task,
                'Metric': 'F1-Score', 'Score': f1_score(y, preds)}
        ]

    results = []
    results.extend(get_metrics(model_qual, X_train_qual,
                   y_train_qual, 'Train', 'Playoff Qualification'))
    results.extend(get_metrics(model_qual, X_eval_qual,
                   y_eval_qual, 'Eval', 'Playoff Qualification'))
    results.extend(get_metrics(model_conf, X_train_conf,
                   y_train_conf, 'Train', 'Conference Finals'))
    results.extend(get_metrics(model_conf, X_eval_conf,
                   y_eval_conf, 'Eval', 'Conference Finals'))

    metrics_df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    tasks = ['Playoff Qualification', 'Conference Finals']
    for i, task in enumerate(tasks):
        sns.barplot(data=metrics_df[metrics_df['Task'] == task], x='Metric',
                    y='Score', hue='Dataset', ax=axes[i], palette='magma')
        axes[i].set_title(f'Generalization: {task}')
        axes[i].set_ylim(0, 1.05)
        axes[i].axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('./eval/logreg/full_model_comparison.png')
    print("Combined performance graph generated.")


# Prepare scaled data and call plot functions
X_eval_qual_scaled = scaler_qual.transform(eval_df[features])
eval_conf_df = eval_df[eval_df["MADE_PLAYOFFS"] == 1].copy()
X_eval_conf_scaled = scaler_conf.transform(eval_conf_df[features])

generate_detailed_eval(
    model_qual, model_conf, features,
    X_train_qual_scaled, y_train_qual, X_eval_qual_scaled, eval_df["MADE_PLAYOFFS"],
    X_train_conf_scaled, y_train_conf, X_eval_conf_scaled, eval_conf_df["MADE_CONF_FINALS"]
)
generate_coefficient_plot(model_qual, model_conf, features)

# --------------------------------------------------
# 9. EXTRA EDA: ROC AND CALIBRATION PLOTS
# --------------------------------------------------
EDA_DIR = Path("eda_logreg")
EDA_DIR.mkdir(exist_ok=True)
eda_df = eval_df.copy()
eda_conf_df = eval_df[eval_df["MADE_PLAYOFFS"] == 1].copy()

if not eda_df.empty and eda_conf_df["MADE_CONF_FINALS"].nunique() >= 2:
    eda_df["PROB_QUAL"] = model_qual.predict_proba(
        scaler_qual.transform(eda_df[features]))[:, 1]
    eda_conf_df["PROB_CONF"] = model_conf.predict_proba(
        scaler_conf.transform(eda_conf_df[features]))[:, 1]

    plt.figure(figsize=(9, 7))
    fpr_q, tpr_q, _ = roc_curve(eda_df["MADE_PLAYOFFS"], eda_df["PROB_QUAL"])
    plt.plot(fpr_q, tpr_q, label=f"Qual (AUC = {auc(fpr_q, tpr_q):.3f})")
    fpr_c, tpr_c, _ = roc_curve(
        eda_conf_df["MADE_CONF_FINALS"], eda_conf_df["PROB_CONF"])
    plt.plot(fpr_c, tpr_c,
             label=f"Conf Finals (AUC = {auc(fpr_c, tpr_c):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curves (Evaluation Block)")
    plt.legend()
    plt.savefig(EDA_DIR / "roc_curves.png")
    plt.close()

    plt.figure(figsize=(9, 7))
    f_q, m_q = calibration_curve(
        eda_df["MADE_PLAYOFFS"], eda_df["PROB_QUAL"], n_bins=5)
    plt.plot(m_q, f_q, "s-", label="Qual")
    f_c, m_c = calibration_curve(
        eda_conf_df["MADE_CONF_FINALS"], eda_conf_df["PROB_CONF"], n_bins=5)
    plt.plot(m_c, f_c, "s-", label="Conf Finals")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Calibration Curves")
    plt.legend()
    plt.savefig(EDA_DIR / "calibration_curves.png")
    plt.close()

print("\nSuccess! Final model trained and results generated.")
