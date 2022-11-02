# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:06:48 2022

@author: ihsan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
#%% Veri İşleme
veri = pd.read_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Veriler/iris.csv")

X = veri.iloc[:,:-1].values
Y = veri.iloc[:,-1].values

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.33, random_state=0)

#%% PCA

pca = PCA(n_components = 2)

X_train2=pca.fit_transform(X_train)
X_test2=pca.fit_transform(X_test)

#%% LDA

lda = LinearDiscriminantAnalysis(n_components = 2)
X_train3 = lda.fit_transform(X_train, y_train)
X_test3 = lda.transform(X_test)

#%% SVC

svc = SVC()
svc_pca = SVC()
svc_lda = SVC()
svc_lda.fit(X_train3, y_train)
y_pred_lda = svc_lda.predict(X_test3)
print(accuracy_score(y_test, y_pred_lda))

p_svc = [{'C':[1,2,3,4,5],'kernel':['linear']},
    {'C':[1,2,3,4,5],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001]},
    {'C':[1,2,3,4,5],'kernel':['poly'],'degree':[1,2,3,4,5],'gamma':[1,0.5,0.1,0.01,0.001]}]

gs = GridSearchCV(estimator = svc, param_grid = p_svc, scoring = "accuracy", cv = 4)
gs_pca = GridSearchCV(estimator = svc_pca, param_grid = p_svc, scoring = "accuracy", cv = 4)
gs_lda = GridSearchCV(estimator = svc_lda, param_grid = p_svc, scoring = "accuracy", cv = 4)

gs_fit = gs.fit(X_train, y_train)
print(gs_fit.best_score_)

gs_pca_fit = gs_pca.fit(X_train2, y_train)
print(gs_pca_fit.best_score_)

gs_lda_fit = gs_lda.fit(X_train3, y_train)
print(gs_lda_fit.best_score_)

"""
kernel = ["linear","rbf","poly"]
gamma = [1,0.5,0.1,0.01,0.001]
for i in range(1,6):
    for k in kernel:
        for j in range(1,8):
            for l in gamma:
                svc_lda = SVC(kernel=k, gamma=l, C = i, degree=j)
                svc_lda.fit(X_train3, y_train)
                y_pred_lda = svc_lda.predict(X_test3)
                print("{} kernel, {} c, {} degree, {} gamma".format(k,i,j,l))
"""       
#%% Logistic Regrssion

log_reg = LogisticRegression(random_state = 0)
log_reg_pca = LogisticRegression(random_state = 0)
log_reg_lda = LogisticRegression(random_state = 0)

log_reg.fit(X_train, y_train)
log_reg_pca.fit(X_train2, y_train)
log_reg_lda.fit(X_train3, y_train)

y_pred_lr = log_reg.predict(X_test)
y_pred_lr_pca = log_reg_pca.predict(X_test2)
y_pred_lr_lda = log_reg_lda.predict(X_test3)

print("LR Acc Score: ", accuracy_score(y_test, y_pred_lr))
print("LR with PCA Acc Score: ", accuracy_score(y_test, y_pred_lr_pca))
print("LR with LDA Acc Score: ", accuracy_score(y_test, y_pred_lr_lda))


"""
ÖDEV: Geri kalanını düzgün bir şekilde yapın!
"""












