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

# Isolate Features and Response
X = df.drop(columns = ["Churn Value"])
Y = df["Churn Value"]
onehot_X = pd.get_dummies(X, drop_first=True)

# Train-Test Split
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    onehot_X, Y, test_size = 0.20, random_state=441
)

# Create Decision Tree
clf = tree.DecisionTreeClassifier(random_state=441)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Impurity vs Alphas
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

# Train a DTree with each alpha
clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=441, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Remove trivial tree
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# Graph number of nodes and depth vs alpha
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

# Compute accuracies
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# Graph accuracy vs alpha
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()

max_test = max(test_scores)
where_max = test_scores.index(max_test)
best_alpha = ccp_alphas[where_max]
print(max_test, where_max, best_alpha)
