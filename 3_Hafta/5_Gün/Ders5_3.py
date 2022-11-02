# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 13:30:05 2022

@author: ihsan
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#veriyi okuma
veri = pd.read_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Veriler/Wine.csv")

X=veri.iloc[:,:13].values
Y=veri.iloc[:,13].values
#eğitim ve test veri ayrımı
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)
#Standartizasasyon
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
#pca
from sklearn.decomposition import PCA
pca=PCA(n_components=2)

X_train2=pca.fit_transform(X_train)
X_test2=pca.fit_transform(X_test)

#PCA olmadan logistic
from sklearn.linear_model import LogisticRegression
log_reg =LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)

#PCA logistic
log_reg_pca=LogisticRegression(random_state=0)
log_reg_pca.fit(X_train2,y_train)

y_pred = log_reg.predict(X_test)
y_pred_pca = log_reg_pca.predict(X_test2)

from sklearn.metrics import confusion_matrix

print("gerçek / PCA'sız")
cm = confusion_matrix(y_test,y_pred)
print(cm)

print("gerçek / pca ile")

cm2= confusion_matrix(y_test, y_pred_pca)
print(cm2)

print("PCA sız ve PCA li")
cm3=confusion_matrix(y_pred, y_pred_pca)

print(cm3)

#pca sız svc
from sklearn.svm import SVC
svc = SVC(kernel="rbf")
svc.fit(X_train,y_train)

#pca svc

svc_pca= SVC(kernel="rbf")
svc_pca.fit(X_train2,y_train)

y_pred_svc=svc.predict(X_test)
y_pred_svc_pca=svc_pca.predict(X_test2)

print("gerçek / PCA'sız")
cm = confusion_matrix(y_test,y_pred_svc)
print(cm)

print("gerçek / pca ile")

cm2= confusion_matrix(y_test, y_pred_svc_pca)
print(cm2)

print("PCA sız ve PCA li")
cm3=confusion_matrix(y_pred_svc, y_pred_svc_pca)

print(cm3)

# pca sız
from sklearn.tree import DecisionTreeClassifier
dtr = DecisionTreeClassifier(criterion = "entropy")
dtr.fit(X_train,y_train)

# pcalı

dtr_pca = DecisionTreeClassifier(criterion = "entropy")
dtr_pca.fit(X_train2,y_train)

y_pred_dtr = dtr.predict(X_test)
y_pred_dtr_pca = dtr_pca.predict(X_test2)

print("gerçek / PCA'sız")
cm = confusion_matrix(y_test,y_pred_dtr)
print(cm)

print("gerçek / pca ile")

cm2= confusion_matrix(y_test, y_pred_dtr_pca)
print(cm2)

print("PCA sız ve PCA li")
cm3=confusion_matrix(y_pred_dtr, y_pred_dtr_pca)

print(cm3)


#%% 

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 10, criterion = "entropy")
# pca sız
rfc.fit(X_train,y_train)

dtr_rfc = RandomForestClassifier(n_estimators = 10, criterion = "entropy")
dtr_rfc.fit(X_train2,y_train)

y_pred_rfc = rfc.predict(X_test)
y_pred_rfc_pca = dtr_rfc.predict(X_test2)

print("gerçek / PCA'sız")
cm = confusion_matrix(y_test,y_pred_rfc)
print(cm)

print("gerçek / pca ile")

cm2= confusion_matrix(y_test, y_pred_rfc_pca)
print(cm2)

print("PCA sız ve PCA li")
cm3=confusion_matrix(y_pred_rfc, y_pred_rfc_pca)

print(cm3)


#%% lda

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components = 2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

#%% LR

log_reg =LogisticRegression(random_state=0)
log_reg.fit(X_train_lda,y_train)

y_pred_lr_lda = log_reg.predict(X_test_lda)

print("Lda ile LR")
cm4 = confusion_matrix(y_pred,  y_pred_lr_lda)
print(cm4)

#%% SVC

svc.fit(X_train_lda,y_train)

y_pred_svc_lda = svc.predict(X_test_lda)

print("LDA ile SVC")
cm_svc = confusion_matrix(y_pred_svc, y_pred_svc_lda)
print(cm_svc)

#%% Decision Tree

dtr.fit(X_train_lda,y_train)

y_pred_dtr_lda = dtr.predict(X_test_lda)

print("DTC ile LDA")
cm_dtc = confusion_matrix(y_pred_dtr, y_pred_dtr_lda)
print(cm_dtc)

#%% RFC

rfc.fit(X_train_lda, y_train)

y_pred_rfc_lda = rfc.predict(X_test_lda)

print("RFC ile LDA")
cm_rfc = confusion_matrix(y_pred_rfc, y_pred_rfc_lda)
print(cm_rfc)





