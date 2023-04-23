##GET DATA 
# smote = True for imbalanced data
ds = Dataset(onehot=True)
X,y = ds.get_training_set()
X_test,y_test = ds.get_testing_set()

###LOGISTIC REGREESSION###
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.utils import resample
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import RFECV
from sklearn import svm

from matplotlib import pyplot

#default, ## change for SMOTE 
lgr = LogisticRegression(solver='liblinear', random_state=441, penalty='l2',C=100)
lgr.fit(X, y)
y_pred = lgr.predict(X_test)

print(ds.accuracy(y_pred))

#cross validation 

all_scrs = cross_val_score(lgr, X, y, cv = 5, scoring='accuracy')

#see the scores 
print('CV score:{}'.format(all_scrs)) 

print('Average score: {:.4f}'.format(all_scrs.mean()))


##BOOTSTRAP 
## to see how the density multiple lengths score

bt_ac = []
bt = 1000
for i in range(bt):
    X_b,y_b = resample(X, y, replace=True)
    y_new = lgr.predict(X_b)
    score = accuracy_score(y_b, y_new)
    bt_ac.append(score)


# plot distribution of accuracy
sns.kdeplot(bt_ac)
plt.title("Accuracy across 1000 bootstrap samples ")
plt.xlabel("Accuracy")
plt.show()


##Performing gridsearchCV, parameters used are fixed using Cross validation methods
parameters = [{'penalty':['l2','l1']}, 
              {'C':[1, 10, 100, 1000]}]

grid_search = GridSearchCV(estimator = lgr,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)

grid_search.fit(X,y )


print('Parameters that give the best results :','\n\n', (grid_search.best_params_))


#RFE similar to best subset but is using recursive elemination. 
rfe_cv = RFECV(estimator=LogisticRegression(max_iter=4000), cv=5, scoring='accuracy')
rfe_cv.fit(X, y)


# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Mean CV")
plt.plot(range(1, len(rfe_cv.cv_results_["mean_test_score"]) + 1), rfe_cv.cv_results_["mean_test_score"])
plt.show()


###confusion matrix 
c = confusion_matrix(y_test, y_pred)
cc_matrix = pd.DataFrame(data=c, columns=['true churn', 'true not churn'], 
                                 index=['predict churn', 'predict not churn''])
sns.heatmap(cc_matrix, annot=True,fmt='d')


imp = lgr.coef_[0]
print(imp)



## Using the best fit lgr use that to test the test set

lgr = LogisticRegression(C=1,solver='liblinear', random_state=441, penalty='l2')

lgr.fit(X, y)
y_pred = lgr.predict(X_test)

print(ds.accuracy(y_pred))

print('Training {:.4f}'.format(lgr.score(X, y)))
print('Test {:.4f}'.format(lgr.score(X_test, y_test)))


#Training set score: 0.8002
#Test set score: 0.8409


y_pred_prob = lgr.predict_proba(X_test)[:, 1]
ds.save_predictions("LGR_pred_prob", y_pred_prob)
