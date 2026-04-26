import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42) #to make testing reproducable
TEST = False
TUNE = False
MAX_DEPTH = 7
MAX_FEATURES = 6
TUNE_SEASON = "2024-25"
PREDICT_SEASON = "2025-26"
TRAINING_DATA = "../data/nba_model_training_data.csv"
STAB_CONST = 1e-10

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
    df = pd.read_csv(TRAINING_DATA)

    print(df.columns)

    feature_cols = [
        "PLUS_MINUS",
        "OFF_RATING_CUSTOM",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "AST",
        "REB",
        "TOV",
        "BLK"
    ]

    if model_no == 0:
        label_col = "MADE_PLAYOFFS"
        model_name = "Playoff Qualification"
    else:
        label_col = "MADE_CONF_FINALS"
        model_name = "Conference Finals"
        df = df[df["MADE_PLAYOFFS"] == 1]

    train_df = df[df["SEASON_ID"] != TUNE_SEASON]
    test_df = df[df["SEASON_ID"] == TUNE_SEASON]


    X_train = train_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    y_train = train_df[label_col].to_numpy()
    y_test = test_df[label_col].to_numpy()

    return df, feature_cols, label_col, X_train, X_test, y_train, y_test, model_name

def run_tuning_depth_vs_features(X_train, X_test, y_train, y_test, model_name):
    total_tune_depth = 10
    total_tune_features = 9

    plt.figure()

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

            print(model_name, "depth:", depth, "max features:", mf, "f1:", score)

        plt.plot(range(2, total_tune_depth + 1), f1_scores, marker="o", label=f"max_features={mf}")

    plt.xlabel("Depth")
    plt.ylabel("F1 Score")
    plt.title(model_name + " F1 vs Max Depth and Max Features")
    plt.legend()
    plt.tight_layout()
    file_name = model_name.replace(" ", "_").lower()
    plt.savefig(f"../eval/forest_output_tuning/{file_name}_depth_features.png", dpi=300)
    plt.close()

def run_tuning_tree_no(X_train, X_test, y_train, y_test, model_name):
    tree_counts = [5, 10, 20, 50, 100, 200]
    f1_scores = []

    depth = MAX_DEPTH
    max_features = int(np.sqrt(X_train.shape[1]))

    for tree_no in tree_counts:
        forest = RandomForest(
            trees_num=tree_no,
            depth=depth,
            features_max=max_features,
        )

        forest.fit(X_train, y_train)
        predictions = forest.predict(X_test)

        score = score_f1(y_test, predictions)
        f1_scores.append(score)

        print(model_name, "tree_number:", tree_no, "f1:", score)

    plt.figure()
    plt.plot(tree_counts, f1_scores, marker="o")
    plt.xlabel("Number of Trees")
    plt.ylabel("F1 Score")
    plt.title(model_name + " F1 vs Number of Trees")
    plt.tight_layout()
    file_name = model_name.replace(" ", "_").lower()
    plt.savefig(f"../eval/forest_output_tuning/{file_name}_tree_no.png", dpi=300)
    plt.close()

def run_tuning(): #used to find the best hyperparameters for run()

    #model 0: Playoffs
    df, feature_cols, label_col, X_train, X_test, y_train, y_test, model_name = tune_setup()

    # -> testing depth_no and feature_no
    run_tuning_depth_vs_features(X_train, X_test, y_train, y_test, model_name)

    # -> testing tree count
    run_tuning_tree_no(X_train, X_test, y_train, y_test, model_name)

    #model 1: Conference Finals
    df, feature_cols, label_col, X_train, X_test, y_train, y_test, model_name = tune_setup(1)

    # -> testing depth_no and feature_no
    run_tuning_depth_vs_features(X_train, X_test, y_train, y_test, model_name)

    # -> testing tree count
    run_tuning_tree_no(X_train, X_test, y_train, y_test, model_name)


def run(): #runs predictions on the model using best hyperparameters
    df = pd.read_csv(TRAINING_DATA)

    feature_cols = [
        "PLUS_MINUS",
        "OFF_RATING_CUSTOM",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "AST",
        "REB",
        "TOV",
        "BLK"
    ]

    train_df = df[df["SEASON_ID"] != PREDICT_SEASON]
    future_df = df[df["SEASON_ID"] == PREDICT_SEASON].copy()

    X_train = train_df[feature_cols].to_numpy()
    X_future = future_df[feature_cols].to_numpy()

    # model 0: PLAYOFFS
    y_train_playoffs = train_df["MADE_PLAYOFFS"].to_numpy()

    playoff_forest = RandomForest(
        trees_num=200,
        depth=MAX_DEPTH,
        features_max=MAX_FEATURES
    )

    playoff_forest.fit(X_train, y_train_playoffs)
    playoff_probs = playoff_forest.predict_proba_1(X_future)

    top16_idx = np.argsort(playoff_probs)[-16:]

    playoff_predictions = [0] * len(X_future)
    for i in top16_idx:
        playoff_predictions[i] = 1

    # model 1: CONFERENCE FINALS
    train_cf_df = train_df[train_df["MADE_PLAYOFFS"] == 1]

    X_train_cf = train_cf_df[feature_cols].to_numpy()
    y_train_cf = train_cf_df["MADE_CONF_FINALS"].to_numpy()

    conf_forest = RandomForest(
        trees_num=200,
        depth=MAX_DEPTH,
        features_max=MAX_FEATURES
    )

    conf_forest.fit(X_train_cf, y_train_cf)

    conf_probs = conf_forest.predict_proba_1(X_future)

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

    output_cols = [
        "TEAM_NAME",
        "SEASON_ID",
        "PRED_MADE_PLAYOFFS",
        "PRED_MADE_CONF_FINALS",
    ]

    print(future_df[output_cols])

    future_df[output_cols].to_csv("../eval/forest_output_predictions/forest_2025_26_predictions.csv", index=False)

if __name__ == "__main__":

    if TEST:
        runTest()
    elif TUNE:
        run_tuning()
    else:
        run()