# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 09:18:25 2022

@author: ihsan
"""

# İmport Library
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import model_selection as ms
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#%% Veri setini yükleme ve hazırlama işlemleri

veri = datasets.load_iris()
# veri = pd.read_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Veriler/iris.csv")

x = veri.data
y = veri.target

x_train, x_test, y_train, y_test = ms.train_test_split(x,y, test_size = 0.33, random_state = 0 )

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#%% SVC model
svc = SVC()

p_svc = [{'C':[1,2,3,4,5],'kernel':['linear']},
    {'C':[1,2,3,4,5],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001]},
    {'C':[1,2,3,4,5],'kernel':['poly'],'degree':[1,2,3,4,5,6,7],'gamma':[1,0.5,0.1,0.01,0.001]}]

grid = GridSearchCV(estimator = svc, param_grid = p_svc, scoring = "accuracy", cv = 4)
grid_search = grid.fit(X_train, y_train)
y_pred_svc = grid_search.predict(X_test)


best_parm_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_

print("Best prameter of gridseach function: ", grid_search.best_params_)
print("Best score of gridsearch function: ",grid_search.best_score_ )

#%% LogisticRegression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state = 0)
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

bas_log = cross_val_score(estimator = logr, X = X_train, y = y_train,cv = 4)
print("Score: ", logr.score(X_test, y_test))
print(bas_log.mean())
print(bas_log.std())

# %% KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1, metric = "minkowski")
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)

bas_knn = cross_val_score(estimator = knn, X = X_train, y = y_train,cv = 4)
print("Score: ", knn.score(X_test, y_test))
print(bas_knn.mean())
print(bas_knn.std())

#%% GridSearch ile KNN

knn_grid = KNeighborsClassifier(metric = "minkowski")

p_knn = {"n_neighbors" : range(1,10), "weights" : ["uniform", "distance"], "p" : [1,2]}

grid_knn = GridSearchCV(estimator = knn_grid, param_grid = p_knn, scoring = "accuracy", cv = 4)
grid_knn_search = grid_knn.fit(X_train, y_train)
y_pred_grid_knn = grid_knn.predict(X_test)

best_parm_grid_knn = grid_knn_search.best_params_
best_score_grid_knn = grid_knn_search.best_score_

print("Best prameter of gridseach function: ", best_parm_grid_knn)
print("Best score of gridsearch function: ",best_score_grid_knn)


#%% naive baise

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_gnb_pred = gnb.predict(X_test)

cm_gnb = confusion_matrix(y_test, y_gnb_pred)
print(cm_gnb)

bas_svc_gnb = cross_val_score(estimator = gnb, X = X_train, y = y_train,cv = 4)
print("Score: ", gnb.score(X_test, y_test))
print(bas_svc_gnb.mean())
print(bas_svc_gnb.std())

#%% decision tree

from sklearn.tree import DecisionTreeClassifier

dtr = DecisionTreeClassifier()

p_dtc = {"criterion" : ["gini", "entropy"], "splitter" : ["best", "random"], }

grid_dtc = GridSearchCV(estimator = dtr, param_grid = p_dtc, scoring = "accuracy", cv = 4)
grid_dtc_search = grid_dtc.fit(X_train, y_train)
y_pred_grid_dtc = grid_dtc.predict(X_test)

best_parm_grid_dtc = grid_dtc_search.best_params_
best_score_grid_dtc = grid_dtc_search.best_score_

print("Best prameter of gridseach function: ", best_parm_grid_dtc)
print("Best score of gridsearch function: ",best_score_grid_dtc)


#%% Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

p_rfc = {"n_estimators" : range(1,50), "criterion" : ["gini", "entropy"], "class_weight" : ["balanced", "balanced_subsample"]}

grid_rfc = GridSearchCV(estimator = rfc, param_grid = p_rfc, scoring = "accuracy", cv = 4)
grid_rfc_search = grid_rfc.fit(X_train, y_train)
y_pred_grid_rfc = grid_rfc.predict(X_test)
    
best_parm_grid_rfc = grid_rfc_search.best_params_
best_score_grid_rfc = grid_rfc_search.best_score_

print("Best prameter of gridseach function: ", best_parm_grid_rfc)
print("Best score of gridsearch function: ",best_score_grid_rfc)

























