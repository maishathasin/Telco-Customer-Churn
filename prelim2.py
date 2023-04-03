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
    "Total Charges"  # Missing values, add back later?
])

# No need to change types to categorical, as one-hot encoding handles that

# Isolate Features and Response
X = df.drop(columns=["Churn Value"])
Y = df["Churn Value"]
onehot_X = pd.get_dummies(X, drop_first=True)

# Split data
n_splits = 5
kf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=441)

# Create fold variables to store data
FoldResults = namedtuple("FoldResults", [
    "ccp_alphas", "impurities", "node_counts", "depth",
    "train_scores", "test_scores"
])
folds = []

# k-fold CV
for train, test in kf.split(onehot_X):
    print("Testing...")
    X_train, X_test = onehot_X.iloc[train], onehot_X.iloc[test]
    y_train, y_test = Y.iloc[train], Y.iloc[test]

    # Create Decision Tree, use Cost Complexity Pruning
    clf = tree.DecisionTreeClassifier(random_state=441)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas[:-1], path.impurities[:-1]

    # Train a DTree with each alpha
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(
            random_state=441, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    # Compute metrics
    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    # Save fold results
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
ax.set_xlabel("Alpha")
ax.set_ylabel("Total Impurity of Terminal Nodes")
ax.set_title(f"Total Impurity vs Alpha for k={n_splits} Folds")
ax.legend()

# Graph number of nodes and depth vs alpha
fig, ax = plt.subplots(2, 1)
for i in range(n_splits):
    ax[0].plot(
        folds[i].ccp_alphas, folds[i].node_counts, marker=",",
        label=f"Fold {i+1}", drawstyle="steps-post"
    )
ax[0].set_xlabel("Alpha")
ax[0].set_ylabel("Number of Nodes")
ax[0].set_title(f"Number of Nodes vs Alpha for k={n_splits} Folds")
ax[0].legend()
for i in range(n_splits):
    ax[1].plot(
        folds[i].ccp_alphas, folds[i].depth, marker=",",
        label=f"Fold {i+1}", drawstyle="steps-post"
    )
ax[1].set_xlabel("Alpha")
ax[1].set_ylabel("Depth")
ax[1].set_title(f"Depth of Classification Tree vs Alpha for k={n_splits} Folds")
ax[1].legend()
fig.tight_layout()

# Graph training accuracy vs alpha
fig, ax = plt.subplots()
ax.set_xlabel("Alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Training Accuracy vs Alpha for k=5 Folds")
for i in range(n_splits):
    ax.plot(
        folds[i].ccp_alphas, folds[i].train_scores, marker=",",
        label=f"Fold {i+1}", drawstyle="steps-post"
    )
ax.legend()

# Graph testing accuracy vs alpha
fig, ax = plt.subplots()
ax.set_xlabel("Alpha")
ax.set_ylabel("Accuracy")
ax.set_title(f"Testing Accuracy vs Alpha for k={n_splits} Folds")
for i in range(n_splits):
    ax.plot(
        folds[i].ccp_alphas, folds[i].test_scores, marker=",",
        label=f"Fold {i+1}", drawstyle="steps-post"
    )
ax.legend()

# Compute average testing accuracy for all alphas
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

# Graph average testing accuracy vs alpha
fig, ax = plt.subplots()
ax.set_xlabel("Alpha")
ax.set_ylabel("Accuracy")
ax.set_title(f"Average Accuracy vs Alpha for k={n_splits} Testing Sets")
ax.plot(
    all_alphas, avg_test_scores, marker=".", drawstyle="steps-post"
)

# Show all plots
plt.show()

# Compute best alpha
max_test = max(avg_test_scores)
where_max = avg_test_scores.index(max_test)
best_alpha = all_alphas[where_max]
print(max_test, where_max, best_alpha)
# Out: 0.79894862168527 1265 0.0006430489816063728
