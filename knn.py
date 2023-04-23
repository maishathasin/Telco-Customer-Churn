##################KNN################################# DEFAULT NO FEATURE ENGINEERING ###############KNN################

######################################KNNNNNN############################################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

##GET DATA ## NO FEATURE ENGINEERING 
ds = Dataset(onehot=True)
X,y = ds.get_training_set()
X_test,y_test = ds.get_testing_set()

knn = KNeighborsClassifier(n_neighbors=15)


# fit the model to the training set, default
knn.fit(X, y)
y_pred = knn.predict(X_test)
print('Accuracy score: {0:0.4f}'. format(ds.accuracy(y_pred)))


##############CV##############################


scores = cross_val_score(knn, X, y, cv = 5, scoring='accuracy')
# average is 0.7843


########GRID SEARCH CV########################


param_grid = param_grid = {'n_neighbors':np.arange(1,50),'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan','cosine']}
               
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)


print(knn_cv.best_score_) #0.784135
print(knn_cv.best_params_)# n_neighbours =14 

y_pred = knn_cv.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(ds.accuracy(y_pred))) ## test on unseen data
##accuracy of 0.7898

#######################SMOTE###########################################################
#Feature engineering 
##CHANGE SCALE = RobustScaler()
##GET DATA
ds = Dataset(onehot=True,scale=True,smote=True)
X,y = ds.get_training_set()
X_test,y_test = ds.get_testing_set()



knn = KNeighborsClassifier(n_neighbors=15)


# fit the model to the training set, default
knn.fit(X, y)
y_pred = knn.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(ds.accuracy(y_pred)))


##############CV##############################


scores = cross_val_score(knn, X, y, cv = 5, scoring='accuracy')
# average is 0.7843


########GRID SEARCH CV########################


param_grid = param_grid = {'n_neighbors':np.arange(1,50),'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan','cosine']}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)


print(knn_cv.best_score_) #0.784135
print(knn_cv.best_params_)# n_neighbours =14 

y_pred = knn_cv.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(ds.accuracy(y_pred))) ## test on unseen data
##accuracy of 0.7898


