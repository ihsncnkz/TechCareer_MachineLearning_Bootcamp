# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 11:42:18 2022

@author: ihsan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

veri = pd.read_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Veriler/veriler.csv")

x = veri.iloc[:,1:4].values
y = veri.iloc[:,4:].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 0 )

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

logr = LogisticRegression(random_state = 0)
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

print(y_pred)
print(y_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

bas_log = cross_val_score(estimator = logr, X = X_train, y = y_train,cv = 4)
print(bas_log.mean())
print(bas_log.std())

# %% KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1, metric = "minkowski")
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(y_pred_knn)
print(y_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)

bas_knn = cross_val_score(estimator = knn, X = X_train, y = y_train,cv = 4)
print(bas_knn.mean())
print(bas_knn.std())

# %% SVM

from sklearn.svm import SVC
svc = SVC(kernel = "rbf")
svc.fit(X_train, y_train)
y_svc_pred = svc.predict(X_test)
print(y_test)
print(y_svc_pred)

cm_svc = confusion_matrix(y_test, y_svc_pred)
print(cm_svc)

bas_svc_rbf = cross_val_score(estimator = svc, X = X_train, y = y_train,cv = 4)
print(bas_svc_rbf.mean())
print(bas_svc_rbf.std())


svc_pol = SVC(kernel = "poly")
svc_pol.fit(X_train, y_train)
y_svc_pol_pred = svc_pol.predict(X_test)

print(y_test)
print(y_svc_pol_pred)

cm_svc_pol = confusion_matrix(y_test, y_svc_pol_pred)
print(cm_svc_pol)

bas_svc_poly = cross_val_score(estimator = svc_pol, X = X_train, y = y_train,cv = 4)
print(bas_svc_poly.mean())
print(bas_svc_poly.std())

svc_lin = SVC(kernel = "linear")
svc_lin.fit(X_train, y_train)
y_svc_lin_pred = svc_lin.predict(X_test)

print(y_test)
print(y_svc_lin_pred)

cm_svc_lin = confusion_matrix(y_test, y_svc_lin_pred)
print(cm_svc_lin)

bas_svc_lin = cross_val_score(estimator = svc_lin, X = X_train, y = y_train,cv = 4)
print(bas_svc_lin.mean())
print(bas_svc_lin.std())


#%% naive baise

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_gnb_pred = gnb.predict(X_test)

print(y_test)
print(y_gnb_pred)

cm_gnb = confusion_matrix(y_test, y_gnb_pred)
print(cm_gnb)

bas_svc_gnb = cross_val_score(estimator = gnb, X = X_train, y = y_train,cv = 4)
print(bas_svc_gnb.mean())
print(bas_svc_gnb.std())

#%% decision tree

from sklearn.tree import DecisionTreeClassifier

dtr = DecisionTreeClassifier(criterion = "entropy")

dtr.fit(X_train, y_train)
y_dtr_pred = dtr.predict(X_test)

print(y_test)
print(y_dtr_pred)

dtr_cm = confusion_matrix(y_test, y_dtr_pred)
print(dtr_cm)

bas_svc_dtr = cross_val_score(estimator = dtr, X = X_train, y = y_train,cv = 4)
print(bas_svc_dtr.mean())
print(bas_svc_dtr.std())

print("************************")

dtr2 = DecisionTreeClassifier(criterion = "gini") 

dtr2.fit(X_train, y_train)
y_dtr_pred2 = dtr2.predict(X_test)

print(y_test)
print(y_dtr_pred2)

dtr_cm2 = confusion_matrix(y_test, y_dtr_pred2)
print(dtr_cm2)

bas_svc_dtr2 = cross_val_score(estimator = dtr2, X = X_train, y = y_train,cv = 4)
print(bas_svc_dtr2.mean())
print(bas_svc_dtr2.std())

#%% Random Forest

from sklearn.ensemble import RandomForestClassifier

for i in range(1,10):
    print("{}. Random Forest result: ".format(i))
    rfc = RandomForestClassifier(n_estimators = i, criterion = "entropy")
    
    rfc.fit(X_train,y_train)
    
    y_pred_rfc = rfc.predict(X_test)
    
    print(y_test)
    print(y_pred_rfc)
    
    rfc_cm = confusion_matrix(y_test, y_pred_rfc)
    print(rfc_cm)

bas_svc_rfc = cross_val_score(estimator = rfc, X = X_train, y = y_train,cv = 4)
print(bas_svc_rfc.mean())
print(bas_svc_rfc.std())

#%% metrics
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

rfc = RandomForestClassifier(n_estimators = 10, criterion = "entropy")

rfc.fit(X_train,y_train)

y_pred_rfc = rfc.predict(X_test)

le = LabelEncoder()
y_test = le.fit_transform(y_test)
y_pred_rfc = le.fit_transform(y_pred_rfc)

fpr, tpr, thold = metrics.roc_curve(y_test, y_pred_rfc, pos_label = 0)

print(fpr)
print(tpr)

#%% GridSearch

from sklearn.model_selection import GridSearchCV

p = [{"C" : [1,2,3,4,5], "kernel" : ["linear"]},
     {"C" : [1,2,3,4,5], "kernel" : ["rbf"], "gamma" : [1,0.5,0.1,0.01,0.001]}]

grid = GridSearchCV(estimator = svc, param_grid = p, scoring = "accuracy", cv = 4)
grid_search = grid.fit(X_train, y_train)

best_parm_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_

print("Best prameter of gridseach function: ", grid_search.best_params_)
print("Best score of gridsearch function: ",grid_search.best_score_ )










