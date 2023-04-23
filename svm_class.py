import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC

# import metrics to compute accuracy
from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report



#############SVM  DEFAULT NO ENgineering###################################################
ds = Dataset(onehot=True)
X,y = ds.get_training_set()
X_test,y_test = ds.get_testing_set()

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

#get probs for AUC
svc=SVC(probability=True) 

parameters = [ {'C':[1, 10, 100, 1000], 'kernel':['linear']},
               {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1,  0.5,  0.9]},
               {'C':[1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.05]} 
              ]

grid_search = GridSearchCV(estimator = svc,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)

svm_cv = grid_search.fit(X, y) 

print(svm_cv.best_params_)

y_pred = svm_cv.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(ds.accuracy(y_pred))) ## test on unseen data

y_pred_prob = svm_cv.predict_proba(X_test)[:, 1]
ds.save_predictions("svm_pred_prob",y_pred_prob)

#############SVM  DEFAULT SMOTE ###################################################

ds = Dataset(onehot=True,scale=True,smote=True)
X,y = ds.get_training_set()
X_test,y_test = ds.get_testing_set()

svc=SVC(probability=True) 

parameters = [ {'C':[1, 10, 100, 1000], 'kernel':['linear']},
               {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1,  0.5,  0.9]},
               {'C':[1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.05]} 
              ]
grid_search = GridSearchCV(estimator = svc,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)
svc_cv = grid_search.fit(X, y) 
print(svc_cv.best_params_)
y_pred = svc_cv.predict(X_test)
## test on unseen data
print('Model accuracy score: {0:0.4f}'. format(ds.accuracy(y_pred))) 
print(classification_report(y_test, y_pred))
###confusion matrix 
c = confusion_matrix(y_test, y_pred)
cc_matrix = pd.DataFrame(data=c, columns=['true churn', 'true not churn'], 
                                 index=['predict churn', 'predict not churn''])
sns.heatmap(cc_matrix, annot=True,fmt='d')

y_pred_prob = svc_cv.predict_proba(X_test)[:, 1]
ds.save_predictions("svm+smote_pred_prob",y_pred_prob)
####bootstrap#########
##BOOTSTRAP 
## to see how the density multiple lengths score

bt_ac = []
bt = 1000
for i in range(bt):
    X_b,y_b = resample(X, y, replace=True)
    y_new = lgr.predict(X_b)
    score = accuracy_score(y_b, y_new)
    bt_ac.append(score)

