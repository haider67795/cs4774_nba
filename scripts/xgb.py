import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# --------------------------------------------------
# 1. DATA LOADING (2004 - 2026 Season Range) [cite: 6, 56]
# --------------------------------------------------
# Ensure these files are in your local directory
try:
    reg_df = pd.read_csv("nba_reg_normalized_r.csv")
    ply_df = pd.read_csv("nba_ply_normalized_r.csv")
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
play_types = ['Isolation','Transition','PRBallHandler','PRRollMan','Postup','Spotup','Handoff','Cut','OffScreen','Misc']
features = [
    "OFF_RATING_CUSTOM",
    "DEF_RATING_CUSTOM",
    #"FG_PCT",
    #"FG3_PCT",
    "FT_PCT",
    "AST_PER_100",
    "REB_PER_100",
    "TOV_PER_100",
    #"PIE_CON_A"
    #"POSS_PCT_Isolation",
    #"POSS_PCT_Transition",
    #"POSS_PCT_PRBallHandler",
    #"POSS_PCT_PRRollMan",
    #"POSS_PCT_Postup",
    #"POSS_PCT_Spotup",
    #"POSS_PCT_Handoff",
    #"POSS_PCT_Cut",
    #"POSS_PCT_OffScreen",
    #"POSS_PCT_Misc"
]
df = df.dropna(subset=features).copy()

# --------------------------------------------------
# 4. TEMPORAL TRAIN/TEST SPLIT [cite: 97]
# --------------------------------------------------
labeled_df = df[df["SEASON_ID"] != "2025-26"].copy()
current_season_df = df[df["SEASON_ID"] == "2025-26"].copy()
train_df, test_df = train_test_split(labeled_df,test_size=0.2, random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'gamma':[1,5,10,20],
    'reg_lambda':[0.1,0.5,0.7,1,2.5,5],
    'reg_alpha':[0.1,0.5,0.7,1,2.5,5]
}

# --------------------------------------------------
# 5. MODEL A: PLAYOFF QUALIFICATION [cite: 27, 82]
# --------------------------------------------------
X_train_qual = train_df[features]
y_train_qual = train_df["MADE_PLAYOFFS"]

scaler_qual = StandardScaler()
X_train_qual_scaled = scaler_qual.fit_transform(X_train_qual)

model_qual = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
#model_qual = LogisticRegression(max_iter=1000, random_state=42)
model_qual.fit(X_train_qual_scaled, y_train_qual)

#results = cross_val_score(model_qual, X_train_qual_scaled, y_train_qual, cv=kfold, scoring='f1')
#print(f"Mean Accuracy: {results.mean():.2f} (Std: {results.std():.2f})")

feat_importances = pd.Series(model_qual.feature_importances_,index=features).astype(float)
feat_importances = feat_importances.sort_values(ascending=False)
print(feat_importances)
plt.bar(feat_importances.index,feat_importances)
plt.xticks(rotation=15)
plt.title('Regular Season Feature Importances')
plt.show()

# --------------------------------------------------
# 6. MODEL B: CONFERENCE FINALS ADVANCEMENT [cite: 27, 74]
# --------------------------------------------------
train_conf_df = train_df[train_df["MADE_PLAYOFFS"] == 1].copy()
X_train_conf = train_conf_df[features]
y_train_conf = train_conf_df["MADE_CONF_FINALS"]

scaler_conf = StandardScaler()
X_train_conf_scaled = scaler_conf.fit_transform(X_train_conf)

model_conf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
#model_conf = XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.1,reg_lambda=1.5,   # L2 regularization
#    reg_alpha=0.5,    # L1 regularization
#    gamma=0.2)
#model_conf = LogisticRegression(max_iter=1000, random_state=42)
model_conf.fit(X_train_conf_scaled, y_train_conf)

#grid_search = RandomizedSearchCV(estimator=model_conf, param_distributions=param_dist, 
                           #scoring='f1', cv=5, verbose=1, n_jobs=-1)
#grid_search.fit(X_train_conf_scaled, y_train_conf)

#print(f"Best Parameters: {grid_search.best_params_}")
#print(f"Best Score: {grid_search.best_score_}")

#results = cross_val_score(model_conf, X_train_conf_scaled, y_train_conf, cv=kfold, scoring='f1')
#print(f"Mean Accuracy: {results.mean():.2f} (Std: {results.std():.2f})")

feat_importances = pd.Series(model_conf.feature_importances_,index=features).astype(float)
feat_importances = feat_importances.sort_values(ascending=False)
print(feat_importances)
plt.bar(feat_importances.index,feat_importances)
plt.xticks(rotation=15)
plt.title('Playoff Feature Importances')
plt.show()

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


# Graph 2: Training Metrics (F1 and Accuracy) [cite: 97]
X_test_qual = test_df[features]
y_test_qual = test_df["MADE_PLAYOFFS"]
test_conf_df = test_df[test_df["MADE_PLAYOFFS"] == 1].copy()
X_test_conf = test_conf_df[features]
y_test_conf = test_conf_df["MADE_CONF_FINALS"]
scaler_qual = StandardScaler()
X_test_qual_scaled = scaler_qual.fit_transform(X_test_qual)
scaler_conf = StandardScaler()
X_test_conf_scaled = scaler_conf.fit_transform(X_test_conf)
train_metrics = {
    'Model': ['Qualification', 'Conf. Finals'],
    'Accuracy': [
        accuracy_score(y_test_qual, model_qual.predict(
            X_test_qual_scaled)),
        accuracy_score(
            y_test_conf, model_conf.predict(X_test_conf_scaled))
    ],
    'F1-Score': [
        f1_score(y_test_qual, model_qual.predict(X_test_qual_scaled)),
        f1_score(y_test_conf, model_conf.predict(X_test_conf_scaled))
    ]
}

perf_df = pd.DataFrame(train_metrics).melt(
    id_vars='Model', var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 6))
sns.barplot(data=perf_df, x='Model', y='Score',
    hue='Metric', palette='magma')
plt.title('XGBoost Accuracy and F1-Score')
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig('model_performance.png')

print(perf_df)