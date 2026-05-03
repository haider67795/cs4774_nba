import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

CONFIG_PATH = "configs/random_forest_config.json"

np.random.seed(42) #to make testing reproducable
TEST = TUNE = PLAYOFF_MAX_DEPTH = PLAYOFF_MAX_FEATURES = TUNE_SEASON_START = PREDICT_SEASON = TRAIN_SEASON_END = \
    TRAINING_DATA = FEATURES_PLAY = FEATURES_CONF = PER_100 = PLAYOFF_TREE_NO = CONF_TREE_NO = CONF_MAX_DEPTH = CONF_MAX_FEATURES = \
    OUTPUT_PRED_DEST = OUTPUT_STAT_DEST = OUTPUT_TUNE_DEST = OUTPUT_COLS =  OUTPUT_TUNE_STATS_DEST = None
STAB_CONST = 1e-10

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def add_per_100_features(df):
    df = df.copy()

    df["AST_PER_100"] = 100 * df["AST"] / (df["POSS"] + STAB_CONST)
    df["REB_PER_100"] = 100 * df["REB"] / (df["POSS"] + STAB_CONST)
    df["TOV_PER_100"] = 100 * df["TOV"] / (df["POSS"] + STAB_CONST)
    df["BLK_PER_100"] = 100 * df["BLK"] / (df["POSS"] + STAB_CONST)

    return df

def setup_mode():
    global TEST, TUNE
    config = load_config()

    TUNE = config["tune"]
    TEST = config["test"]


def setup_model():
    global TEST, TUNE, PLAYOFF_MAX_DEPTH, PLAYOFF_MAX_FEATURES, PREDICT_SEASON, TUNE_SEASON_START, TRAIN_SEASON_END, \
        TRAINING_DATA, PER_100, FEATURES_PLAY, FEATURES_CONF, PLAYOFF_TREE_NO, CONF_TREE_NO, \
        OUTPUT_PRED_DEST, OUTPUT_STAT_DEST, OUTPUT_TUNE_DEST, OUTPUT_COLS, OUTPUT_TUNE_STATS_DEST, \
        CONF_MAX_DEPTH, CONF_MAX_FEATURES


    config = load_config()
    df = pd.read_csv(config["training_data"])

    PER_100 = config["use_per_100"]
    PLAYOFF_MAX_DEPTH = config["playoff_max_depth"]
    CONF_MAX_DEPTH = config["conf_max_depth"]
    PLAYOFF_MAX_FEATURES = config["playoff_max_features"]
    CONF_MAX_FEATURES = config["conf_max_features"]
    PREDICT_SEASON = config["predict_season"]
    TUNE_SEASON_START = config["tune_season_start"]
    TRAIN_SEASON_END = config["train_season_end"]
    TRAINING_DATA = config["training_data"]
    PLAYOFF_TREE_NO = config["playoff_trees"]
    CONF_TREE_NO = config["conference_trees"]
    OUTPUT_PRED_DEST = config["output_pred_dest"]
    OUTPUT_STAT_DEST = config["output_stats_dest"]
    OUTPUT_TUNE_DEST = config["output_tune_dest"]
    OUTPUT_COLS = config["output_cols"]
    OUTPUT_TUNE_STATS_DEST = config["output_tune_stats_dest"]

    if PER_100:
        df = add_per_100_features(df)
        FEATURES_PLAY = config["per_100_features_playoff"]
        FEATURES_CONF = config["per_100_features_conf"]
    else:
        FEATURES_PLAY = config["raw_features_playoff"]
        FEATURES_CONF = config["raw_features_conf"]

    return df

def create_output_dirs():
    os.makedirs(OUTPUT_TUNE_DEST, exist_ok=True)
    os.makedirs(OUTPUT_TUNE_STATS_DEST, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PRED_DEST), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_STAT_DEST), exist_ok=True)


def bootstrap_bounds(x, y):
    n = len(x)
    indices = np.random.randint(0, n, size=n)
    return x[indices], y[indices]

def maj_vote(votes):
    val, count = np.unique(votes, return_counts=True)
    return int(val[np.argmax(count)])

class DecisionTree:

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, vote=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.vote = vote



    def __init__(self, max_depth=10, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None
        self.features_num = None

    def make_vote(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def get_entropy(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return -np.sum((counts / len(y))*np.log2((counts / len(y))+STAB_CONST))

    def find_split(self,X,y):
        max_gain = float('-inf')
        split_ftr = None
        split_thresh = None
        split_left = None
        split_right = None
        n = len(y)

        features = np.random.choice(X.shape[1],self.max_features,replace=False)

        p_ent = self.get_entropy(y)

        for ftr in features:
            thresholds = np.unique(X[:,ftr])

            for thresh in thresholds:
                left = X[:,ftr] < thresh
                right = X[:,ftr] >= thresh

                if len(y[left]) > 0  and len(y[right]) > 0:
                    l_ent = self.get_entropy(y[left])
                    r_ent = self.get_entropy(y[right])


                    ent = len(y[left]) / n * l_ent + len(y[right]) / n * r_ent

                    gain = p_ent - ent

                    if gain > max_gain:
                        max_gain = gain
                        split_ftr = ftr
                        split_thresh = thresh
                        split_left = left
                        split_right = right

        return split_ftr, split_thresh, split_left, split_right



    def build(self, X, y, depth):

        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return self.Node(vote=self.make_vote(y))

        split_ftr, split_thresh, split_left, split_right = self.find_split(X, y)

        if split_ftr is None:
            return self.Node(vote=self.make_vote(y))

        left = self.build(X[split_left], y[split_left], depth + 1)
        right = self.build(X[split_right], y[split_right], depth + 1)

        return self.Node(feature=split_ftr, threshold=split_thresh, left=left, right=right)

    def fit(self, X, y):
        self.features_num = X.shape[1]
        depth = 0

        if self.max_features is None:
            self.max_features = int(np.sqrt(self.features_num))

        self.root = self.build(X, y, depth)

    def predict_one(self, row):
        node = self.root

        while node.vote is None:
            if row[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.vote

    def predict(self, X):
        return np.array([self.predict_one(row) for row in X])


class RandomForest:
    def __init__(self, trees_num, depth, features_max=None):
        self.trees_num = trees_num
        self.depth = depth
        self.features_max = features_max
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        features_num = X.shape[1]

        if self.features_max is None:
            self.features_max = int(np.sqrt(features_num))

        for i in range(self.trees_num):
            x_bootstrap, y_bootstrap = bootstrap_bounds(X, y)

            tree = DecisionTree(max_depth=self.depth, max_features=self.features_max)
            tree.fit(x_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = []

        for tree in self.trees:
            predictions.append(tree.predict(X))

        result = []
        for i in range(len(X)):
            votes = []
            for t in range(self.trees_num):
                votes.append(predictions[t][i])
            result.append(maj_vote(votes))

        return result

    def predict_proba_1(self, X):
        predictions = []

        for tree in self.trees:
            predictions.append(tree.predict(X))

        predictions = np.array(predictions)

        return np.mean(predictions == 1, axis=0)

def score_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

def score_f1(y_true, y_pred):
    tp = fp = fn = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1

    precision = tp / (tp + fp + STAB_CONST)
    recall = tp / (tp + fn + STAB_CONST)

    return 2 * (precision * recall) / (precision + recall + STAB_CONST)

def runTest():
    X = np.array([
        [1], [2], [3], [4], [5]
    ])
    y = np.array([0,0,0,1,1])

    print("Running Test on Decision Tree: ")
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)

    tree_predictions = tree.predict(X)
    print("Tree predictions: ", tree_predictions)
    print("True labels: ", y)

    print("\nTesting on Random Forest: ")
    forest = RandomForest(trees_num=10, depth=3)
    forest.fit(X, y)

    forest_predictions = forest.predict(X)
    print("Forest predictions: ", forest_predictions)
    print("True labels: ", y)

def tune_setup(model_no=0):
    df = setup_model()
    create_output_dirs()

    print(df.columns)

    if model_no == 0:
        label_col = "MADE_PLAYOFFS"
        model_name = "Playoff Qualification"
    else:
        label_col = "MADE_CONF_FINALS"
        model_name = "Conference Finals"
        df = df[df["MADE_PLAYOFFS"] == 1]

    train_df = df[df["SEASON_ID"] <= TRAIN_SEASON_END]

    test_df = df[
        (df["SEASON_ID"] >= TUNE_SEASON_START) &
        (df["SEASON_ID"] != PREDICT_SEASON)
        ]

    X_train = train_df[FEATURES_PLAY].to_numpy()
    X_test = test_df[FEATURES_PLAY].to_numpy()

    y_train = train_df[label_col].to_numpy()
    y_test = test_df[label_col].to_numpy()

    return df, label_col, X_train, X_test, y_train, y_test, model_name

def run_tune(X_train, X_test, y_train, y_test, model_name):
    total_tune_depth = 10
    total_tune_features = 6
    tree_counts = [5, 10, 20, 50, 100, 200]

    plt.figure()
    rows = []

    for trees in tree_counts:
        for mf in range(2, total_tune_features + 1):
            f1_scores = []

            for depth in range(2, total_tune_depth + 1):
                forest = RandomForest(trees_num=trees, depth=depth, features_max=mf)

                forest.fit(X_train, y_train)
                predictions = forest.predict(X_test)

                print("Predicted positives:", sum(predictions))
                print("Actual positives:", sum(y_test))
                print("Predictions:", predictions)
                print("Actual:", list(y_test))

                score = score_f1(y_test, predictions)
                f1_scores.append(score)

                rows.append({
                    "model": model_name,
                    "max_depth": depth,
                    "max_features": mf,
                    "trees": trees,
                    "f1": score,
                    "actual_positives": int(sum(y_test)),
                    "predicted_positives": int(sum(predictions))
                })

                print(model_name, "depth:", depth, "max features:", mf, "f1:", score)

            plt.plot(range(2, total_tune_depth + 1), f1_scores, marker="o", label=f"max_features={mf}")

    plt.xlabel("Depth")
    plt.ylabel("F1 Score")
    plt.title(model_name + " F1 vs Max Depth and Max Features")
    plt.legend()
    plt.tight_layout()
    file_name = model_name.replace(" ", "_").lower()
    plt.savefig(f"{OUTPUT_TUNE_DEST}/{file_name}_depth_features.png", dpi=300)
    plt.close()

    pd.DataFrame(rows).to_csv(
        f"{OUTPUT_TUNE_STATS_DEST}/{file_name}_depth_features_stats.csv",
        index=False
    )


def run_tuning_depth_vs_features(X_train, X_test, y_train, y_test, model_name):
    total_tune_depth = 10
    total_tune_features = 6

    plt.figure()
    rows = []

    for mf in range(2, total_tune_features + 1):
        f1_scores = []

        for depth in range(2, total_tune_depth + 1):
            forest = RandomForest(trees_num=50, depth=depth, features_max=mf)

            forest.fit(X_train, y_train)
            predictions = forest.predict(X_test)

            print("Predicted positives:", sum(predictions))
            print("Actual positives:", sum(y_test))
            print("Predictions:", predictions)
            print("Actual:", list(y_test))

            score = score_f1(y_test, predictions)
            f1_scores.append(score)

            rows.append({
                "model": model_name,
                "max_depth": depth,
                "max_features": mf,
                "trees": 50,
                "f1": score,
                "actual_positives": int(sum(y_test)),
                "predicted_positives": int(sum(predictions))
            })

            print(model_name, "depth:", depth, "max features:", mf, "f1:", score)

        plt.plot(range(2, total_tune_depth + 1), f1_scores, marker="o", label=f"max_features={mf}")

    plt.xlabel("Depth")
    plt.ylabel("F1 Score")
    plt.title(model_name + " F1 vs Max Depth and Max Features")
    plt.legend()
    plt.tight_layout()
    file_name = model_name.replace(" ", "_").lower()
    plt.savefig(f"{OUTPUT_TUNE_DEST}/{file_name}_depth_features.png", dpi=300)
    plt.close()

    pd.DataFrame(rows).to_csv(
        f"{OUTPUT_TUNE_STATS_DEST}/{file_name}_depth_features_stats.csv",
        index=False
    )

def run_tuning_tree_no(X_train, X_test, y_train, y_test, model_name):
    tree_counts = [5, 10, 20, 50, 100, 200]
    f1_scores = []

    max_features = int(np.sqrt(X_train.shape[1]))

    rows = []
    if model_name == "Playoff Qualification":
        max_depth = PLAYOFF_MAX_DEPTH
    else:
        max_depth = CONF_MAX_DEPTH
    for tree_no in tree_counts:
        forest = RandomForest(
            trees_num=tree_no,
            depth=PLAYOFF_MAX_DEPTH,
            features_max=max_features,
        )

        forest.fit(X_train, y_train)
        predictions = forest.predict(X_test)

        score = score_f1(y_test, predictions)
        f1_scores.append(score)

        rows.append({
            "model": model_name,

            "max_features": max_features,
            "trees": tree_no,
            "f1": score,
            "actual_positives": int(sum(y_test)),
            "predicted_positives": int(sum(predictions))
        })

        print(model_name, "tree_number:", tree_no, "f1:", score)

    plt.figure()
    plt.plot(tree_counts, f1_scores, marker="o")
    plt.xlabel("Number of Trees")
    plt.ylabel("F1 Score")
    plt.title(model_name + " F1 vs Number of Trees")
    plt.tight_layout()
    file_name = model_name.replace(" ", "_").lower()
    plt.savefig(f"{OUTPUT_TUNE_DEST}/{file_name}_tree_no.png", dpi=300)
    plt.close()

    pd.DataFrame(rows).to_csv(
        f"{OUTPUT_TUNE_STATS_DEST}/{file_name}_tree_no_stats.csv",
        index=False
    )

def run_tuning(): #used to find the best hyperparameters for run()

    #model 0: Playoffs
    df, label_col, X_train, X_test, y_train, y_test, model_name = tune_setup()

    # -> testing depth_no and feature_no
   # run_tuning_depth_vs_features(X_train, X_test, y_train, y_test, model_name)

    # -> testing tree count
   # run_tuning_tree_no(X_train, X_test, y_train, y_test, model_name)

    run_tune(X_train, X_test, y_train, y_test, model_name)

    #model 1: Conference Finals
    df, label_col, X_train, X_test, y_train, y_test, model_name = tune_setup(1)

    # -> testing depth_no and feature_no
   # run_tuning_depth_vs_features(X_train, X_test, y_train, y_test, model_name)

    # -> testing tree count
   # run_tuning_tree_no(X_train, X_test, y_train, y_test, model_name)

    run_tune(X_train, X_test, y_train, y_test, model_name)


def plot_rf_bar(playoff_acc, playoff_f1, conf_acc, conf_f1):
    models = ["Qualification", "Conf. Finals"]
    accuracy_scores = [playoff_acc, conf_acc]
    f1_scores = [playoff_f1, conf_f1]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(x - width / 2, accuracy_scores, width, label="Accuracy", color="#6f2c7f")
    plt.bar(x + width / 2, f1_scores, width, label="F1-Score", color="#df6f6f")


    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.title("Random Forest Accuracy and F1-Score")
    plt.xticks(x, models)
    plt.legend(title="Metric")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_TUNE_DEST}/rf_summary_bar.png", dpi=300)
    plt.close()

def run():
    df = setup_model()
    create_output_dirs()

    train_df = df[df["SEASON_ID"] <= TRAIN_SEASON_END]

    test_df = df[
        (df["SEASON_ID"] > TRAIN_SEASON_END) &
        (df["SEASON_ID"] != PREDICT_SEASON)
    ]

    X_train = train_df[FEATURES_PLAY].to_numpy()
    X_test = test_df[FEATURES_PLAY].to_numpy()

    y_train_playoffs = train_df["MADE_PLAYOFFS"].to_numpy()

    playoff_forest = RandomForest(
        trees_num=PLAYOFF_TREE_NO,
        depth=PLAYOFF_MAX_DEPTH,
        features_max=PLAYOFF_MAX_FEATURES
    )

    playoff_forest.fit(X_train, y_train_playoffs)

    y_true_playoff = test_df["MADE_PLAYOFFS"].to_numpy()
    y_pred_playoff = playoff_forest.predict(X_test)

    playoff_acc = score_accuracy(y_true_playoff, y_pred_playoff)
    playoff_f1 = score_f1(y_true_playoff, y_pred_playoff)

    train_df = train_df.copy()
    train_df["PLAYOFF_PROB"] = playoff_forest.predict_proba_1(X_train)

    train_cf_df = train_df[train_df["MADE_PLAYOFFS"] == 1].copy()

    X_train_cf = train_cf_df[FEATURES_CONF].to_numpy()
    y_train_cf = train_cf_df["MADE_CONF_FINALS"].to_numpy()

    conf_forest = RandomForest(
        trees_num=CONF_TREE_NO,
        depth=CONF_MAX_DEPTH,
        features_max=CONF_MAX_FEATURES
    )

    conf_forest.fit(X_train_cf, y_train_cf)

    test_df = test_df.copy()
    test_df["PLAYOFF_PROB"] = playoff_forest.predict_proba_1(X_test)
    test_df["PRED_MADE_PLAYOFFS"] = 0

    for season in test_df["SEASON_ID"].unique():
        season_mask = test_df["SEASON_ID"] == season
        season_indices = np.where(season_mask)[0]

        season_probs = test_df.loc[season_mask, "PLAYOFF_PROB"].to_numpy()
        top16_local = season_indices[np.argsort(season_probs)[-16:]]

        test_df.iloc[
            top16_local,
            test_df.columns.get_loc("PRED_MADE_PLAYOFFS")
        ] = 1

    test_cf_df = test_df[test_df["PRED_MADE_PLAYOFFS"] == 1].copy()

    X_test_cf = test_cf_df[FEATURES_CONF].to_numpy()
    y_true_conf = test_cf_df["MADE_CONF_FINALS"].to_numpy()

    conf_test_probs = conf_forest.predict_proba_1(X_test_cf)
    test_cf_df["CONF_PROB"] = conf_test_probs

    y_pred_conf = np.zeros(len(test_cf_df), dtype=int)

    for season in test_cf_df["SEASON_ID"].unique():
        season_mask = test_cf_df["SEASON_ID"] == season
        season_indices = np.where(season_mask)[0]

        season_probs = conf_test_probs[season_indices]
        top4_local = season_indices[np.argsort(season_probs)[-4:]]

        y_pred_conf[top4_local] = 1

    conf_acc = score_accuracy(y_true_conf, y_pred_conf)
    conf_f1 = score_f1(y_true_conf, y_pred_conf)

    print("Random Forest Evaluation")
    print("Playoff Qualification -> Accuracy:", playoff_acc, "F1:", playoff_f1)
    print("Conference Finals -> Accuracy:", conf_acc, "F1:", conf_f1)

    plot_rf_bar(playoff_acc, playoff_f1, conf_acc, conf_f1)

    full_train_df = df[df["SEASON_ID"] != PREDICT_SEASON].copy()
    future_df = df[df["SEASON_ID"] == PREDICT_SEASON].copy()

    X_full_train = full_train_df[FEATURES_PLAY].to_numpy()
    X_future = future_df[FEATURES_PLAY].to_numpy()

    y_full_playoffs = full_train_df["MADE_PLAYOFFS"].to_numpy()

    playoff_forest = RandomForest(
        trees_num=PLAYOFF_TREE_NO,
        depth=PLAYOFF_MAX_DEPTH,
        features_max=PLAYOFF_MAX_FEATURES
    )

    playoff_forest.fit(X_full_train, y_full_playoffs)

    playoff_probs = playoff_forest.predict_proba_1(X_future)

    top16_idx = np.argsort(playoff_probs)[-16:]

    playoff_predictions = [0] * len(X_future)
    for i in top16_idx:
        playoff_predictions[i] = 1

    full_train_df["PLAYOFF_PROB"] = playoff_forest.predict_proba_1(X_full_train)

    full_train_cf_df = full_train_df[full_train_df["MADE_PLAYOFFS"] == 1].copy()

    X_full_train_cf = full_train_cf_df[FEATURES_CONF].to_numpy()
    y_full_train_cf = full_train_cf_df["MADE_CONF_FINALS"].to_numpy()

    conf_forest = RandomForest(
        trees_num=CONF_TREE_NO,
        depth=CONF_MAX_DEPTH,
        features_max=CONF_MAX_FEATURES
    )

    conf_forest.fit(X_full_train_cf, y_full_train_cf)

    future_df["PLAYOFF_PROB"] = playoff_probs

    X_future_cf = future_df[FEATURES_CONF].to_numpy()
    conf_probs = conf_forest.predict_proba_1(X_future_cf)

    eligible_indices = [
        i for i in range(len(X_future))
        if playoff_predictions[i] == 1
    ]

    eligible_indices_sorted = sorted(
        eligible_indices,
        key=lambda i: conf_probs[i],
        reverse=True
    )

    top4_indices = eligible_indices_sorted[:4]

    conf_predictions = [0] * len(X_future)
    for i in top4_indices:
        conf_predictions[i] = 1

    future_df["PRED_MADE_PLAYOFFS"] = playoff_predictions
    future_df["PRED_MADE_CONF_FINALS"] = conf_predictions
    future_df["PRED_PLAYOFF_PROB"] = playoff_probs
    future_df["PRED_CONF_FINALS_PROB"] = conf_probs

    print(future_df[OUTPUT_COLS])

    future_df[OUTPUT_COLS].to_csv(OUTPUT_PRED_DEST, index=False)

    stats = {
        "predict_season": PREDICT_SEASON,
        "use_per_100": PER_100,
        "playoff_max_depth": PLAYOFF_MAX_DEPTH,
        "playoff_max_features": PLAYOFF_MAX_FEATURES,
        "conf_max_depth": CONF_MAX_DEPTH,
        "conf_max_features": CONF_MAX_FEATURES,
        "playoff_trees": PLAYOFF_TREE_NO,
        "conference_trees": CONF_TREE_NO,
        "playoff_accuracy": playoff_acc,
        "playoff_f1": playoff_f1,
        "conference_accuracy": conf_acc,
        "conference_f1": conf_f1,
        "predicted_playoff_teams": int(sum(playoff_predictions)),
        "predicted_conference_finalists": int(sum(conf_predictions))
    }

    pd.DataFrame([stats]).to_csv(OUTPUT_STAT_DEST, index=False)

if __name__ == "__main__":

    setup_mode()

    if TEST:
        runTest()
    elif TUNE:
        run_tuning()
    else:
        run()