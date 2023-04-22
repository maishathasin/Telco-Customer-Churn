from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
import seaborn as sns


lgr = LogisticRegression(solver='liblinear', random_state=441)

lgr.fit(X, y)
y_pred = lgr.predict(X_test)


## Default Model accuracy score 0.8438
print('Accuracy score:'. format(ds.accuracy(y_pred)))



parameters = [{'penalty':['l1','l2']}, 
              {'C':[0.001,0.1,1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = lgr,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10, #Cross validation 
                           verbose=0)


grid_search.fit(X, y)

#GridSearch CV score on test set: 0.8509


#cross validation 

scores = cross_val_score(lgr, X, y, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))


print('Average cross-validation score:'(scores.mean()))



## trying with bootstrap, SMOTE oversample + Kfold  + GridsearchCV for best parameters 
accuracy = []
it = 1000
for i in range(it):
    X_b,y_b = resample(X, y, replace=True)
    y_new = logreg.predict(X_b)
    score = accuracy_score(y_new, y_new)
    accuracy.append(score)


# plot distribution of accuracy
sns.kdeplot(accuracy)
plt.title("Accuracy across 1000 bootstrap samples of the held-out test set")
plt.xlabel("Accuracy")
plt.show()

## Peak at around 0.8333


#todo: get test set from telco.py

#precision    recall  f1-score   support

         #  0       0.85      0.88      0.86       522
          # 1       0.87      0.84      0.85       511


#ROC AUC 0.85 
