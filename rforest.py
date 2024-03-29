from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from telco import Dataset


def create_random_forest(load=True, smote=False, save=False):
    """
    Given a dataset, tunes or loads a 100-tree random forest
    """
    ds = Dataset(onehot=True, scale=True, smote=smote)

    if not load:
        param_grid = [{
            'max_depth': list(range(1, 16))
        }]
        clf = GridSearchCV(
            RandomForestClassifier(
            oob_score=True, n_jobs=-1, random_state=441
        ),
            param_grid, cv=5, scoring=['accuracy', 'f1'], refit='accuracy'
        )
    elif smote:
        # Loads the best parameters from the search for smote=True
        clf = RandomForestClassifier(
            max_depth=11, oob_score=True, n_jobs=-1, random_state=441
        )
    else:
        # Loads the best parameters from the search for smote=False
        clf = RandomForestClassifier(
            max_depth=8, oob_score=True, n_jobs=-1, random_state=441
        )

    clf.fit(*ds.get_training_set())

    if not load:
        print(clf.best_params_)

    y_pred = clf.predict(ds.get_testing_set())
    y_pred_prob = clf.predict_proba(ds.get_testing_set())[:, 1]
    acc = ds.accuracy(y_pred)
    f1 = ds.f1(y_pred)

    ds.report(y_pred)
    ds.heatmap(y_pred)

    if save:
        ds.save_predictions("rf+sm" if smote else "rf", y_pred_prob)
    
    return (round(acc, 4), round(f1, 4))


print(create_random_forest(load=True, smote=False, save=True))
print(create_random_forest(load=True, smote=True, save=True))
