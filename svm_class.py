import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC

# import metrics to compute accuracy
from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report




df = pd.read_excel('/content/Telco_customer_churn.xlsx')
df.head()


df.columns = df.columns.str.replace(' ', '')
df['Churn Label'].value_counts()/np.float(len(df))



dd  = df.loc[df['TotalCharges'] != ' '] # removing rows with no total charges
dd['TotalCharges'] = dd['TotalCharges'].astype('float')
X = dd.drop(['ChurnLabel','CustomerID','Count','Country','State','City','ZipCode','LatLong','Latitude','Longitude','ChurnValue','ChurnScore','CLTV','ChurnReason'], axis=1)
#one hot encoding
dd['ChurnLabel'] = dd['ChurnLabel'].replace(to_replace = ['Yes','No'],value = ['1','0'])
X = pd.get_dummies(X)
y = dd['ChurnLabel']



#split train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
X_train.shape, X_test.shape




svc=SVC() 
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

svc=SVC(C=100)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Model accuracy score with C = 100: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

svc=SVC(kernel='poly',C=100)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print('Model accuracy score with C = 100: {0:0.4f} and polynomial kernal'. format(accuracy_score(y_test, y_pred)))

svc=SVC(kernel='sigmoid',C=100)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Model accuracy score with C = 100: {0:0.4f} and sigmoid kernal'. format(accuracy_score(y_test, y_pred)))


#Model accuracy score with default hyperparameters: 0.7585
#Model accuracy score with C = 100: 0.8011
#Model accuracy score with C = 100: 0.7585 and polynomial kernal
#Model accuracy score with C = 100: 0.6264 and sigmoid kernal
