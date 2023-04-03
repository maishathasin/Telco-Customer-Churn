from collections import namedtuple

import matplotlib.pyplot as plt
from sklearn import model_selection, tree
import pandas as pd


# Load data
df = pd.read_excel("Telco_customer_churn.xlsx")
df = df.drop(columns=[
    "CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long",
    "Latitude", "Longitude", "Churn Label", "Churn Score", "CLTV",
    "Churn Reason",
    "Total Charges" # Missing values, add back later
])

# Set types
df = df.astype("category")
df["Tenure Months"] = df["Tenure Months"].astype("int64")
df["Monthly Charges"] = df["Monthly Charges"].astype("float64")

# Isolate Features and Response
X = df.drop(columns = ["Churn Value"])
Y = df["Churn Value"]
onehot_X = pd.get_dummies(X, drop_first=True)

# k-fold CV
n_splits = 5
kf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=441)

# Create fold variables to store data
FoldResults = namedtuple("FoldResults", [
    "ccp_alphas", "impurities", "node_counts", "depth", 
    "train_scores", "test_scores"
])
folds = []

for train, test in kf.split(onehot_X):
    print("Testing...")
    X_train, X_test = onehot_X.iloc[train], onehot_X.iloc[test]
    y_train, y_test = Y.iloc[train], Y.iloc[test]

    # Create Decision Tree
    clf = tree.DecisionTreeClassifier(random_state=441)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas[:-1], path.impurities[:-1]

    # Train a DTree with each alpha
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=441, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    # Compute metrics
    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    folds.append(FoldResults(
        ccp_alphas, impurities, node_counts, depth, train_scores, test_scores
    ))

# Impurity vs Alphas
fig, ax = plt.subplots()
for i in range(n_splits):
    ax.plot(
        folds[i].ccp_alphas, folds[i].impurities, marker=",",
        label=f"Fold {i+1}", drawstyle="steps-post"
    )
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

# Graph number of nodes and depth vs alpha
fig, ax = plt.subplots(2, 1)
for i in range(n_splits):
    ax[0].plot(
        folds[i].ccp_alphas, folds[i].node_counts, marker=",",
        label=f"Fold {i+1}", drawstyle="steps-post"
    )
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
for i in range(n_splits):
    ax[1].plot(
        folds[i].ccp_alphas, folds[i].depth, marker=",",
        label=f"Fold {i+1}", drawstyle="steps-post"
    )
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

for i in range(n_splits):
    print(folds[i].ccp_alphas[:10])

# Graph accuracy vs alpha
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training sets")
for i in range(n_splits):
    ax.plot(
        folds[i].ccp_alphas, folds[i].train_scores, marker=",",
        label=f"Fold {i+1}", drawstyle="steps-post"
    )
ax.legend()

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for testing sets")
for i in range(n_splits):
    ax.plot(
        folds[i].ccp_alphas, folds[i].test_scores, marker=",",
        label=f"Fold {i+1}", drawstyle="steps-post"
    )
ax.legend()

all_alphas = sorted(set(
    sum([list(folds[i].ccp_alphas) for i in range(n_splits)], [])
))

avg_test_scores = []
a_ind = [0 for _ in range(n_splits)]
for alpha in all_alphas:
    a_sum = 0
    for i in range(n_splits):
        while a_ind[i] + 1 < len(folds[i].ccp_alphas):
            if folds[i].ccp_alphas[a_ind[i] + 1] < alpha:
                a_ind[i] += 1
            else:
                break
        a_sum += folds[i].test_scores[a_ind[i]]
    avg_test_scores.append(a_sum/n_splits)
        
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Average Accuracy vs alpha for testing sets")
ax.plot(
    all_alphas, avg_test_scores, marker=".", drawstyle="steps-post"
)
ax.legend()

plt.show()

# max_test = max(test_scores)
# where_max = test_scores.index(max_test)
# best_alpha = ccp_alphas[where_max]
# print(max_test, where_max, best_alpha)
