# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:57:31 2022

@author: ihsan
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mt
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import activations

veri = pd.read_excel("E:/Work/Bootcamp/TechCareer/Machine_Learning/Ders6/sondata.xlsx")

x = veri.iloc[:,:10].values
y = veri.iloc[:,10].values

sc = pr.StandardScaler()
X = sc.fit_transform(x)
y = pr.LabelEncoder().fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)

#%% Logistic regression
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)

y_pred1 = lr.predict(x_test)

cross_lr = cross_val_score(estimator = lr, X = x_train, y = y_train, cv = 4)
y_pred2_lr_acc = cross_lr.mean()


doğ_değ_lr = mt.confusion_matrix(y_test, y_pred1)
y_pred1_lr_acc = mt.accuracy_score(y_test, y_pred1)


#%% Naive bayes

bnb = BernoulliNB()
bnb.fit(x_train, y_train)
y_pred_bnb = bnb.predict(x_test)

cross_bnb = cross_val_score(estimator = bnb, X=x_train, y = y_train, cv = 4)
y_pred2_bnb = cross_bnb.mean()

dog_değ_bnb = mt.confusion_matrix(y_test, y_pred_bnb)
y_pred1_bnb_acc = mt.accuracy_score(y_test, y_pred_bnb)

#%% Dtc

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_pred_dtc = dtc.predict(x_test)

cross_dtc = cross_val_score(estimator = dtc, X = x_train, y = y_train, cv = 4)
y_pred2_dtc_acc = cross_dtc.mean()

dog_değ_dtc = mt.confusion_matrix(y_test, y_pred_dtc)
y_pred1_dtc_acc = mt.accuracy_score(y_test, y_pred_dtc)

#%% Random forest

rfc = RandomForestClassifier(n_estimators = 10, criterion = "entropy")
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)

cross_rfc = cross_val_score(estimator = rfc, X=x_train, y = y_train, cv = 4)
y_pred2_rfc_acc = cross_rfc.mean()

dog_değ_rfc = mt.confusion_matrix(y_test, y_pred_rfc)
y_pred1_rfc_acc = mt.accuracy_score(y_test, y_pred_rfc)

#%% knn

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)

cross_knn = cross_val_score(estimator = knn, X=x_train, y = y_train, cv = 4)
y_pred2_knn_acc = cross_knn.mean()

dog_değ_knn = mt.confusion_matrix(y_test, y_pred_knn)
y_pred1_knn_acc = mt.accuracy_score(y_test, y_pred_knn)

#%% SVC

svc = SVC(kernel="rbf")
svc.fit(x_train,y_train)
y_pred_svc = svc.predict(x_test)

cross_svc = cross_val_score(estimator = svc, X=x_train, y = y_train, cv = 4)
y_pred2_svc_acc = cross_svc.mean()

dog_değ_svc = mt.confusion_matrix(y_test, y_pred_svc)
y_pred1_svc_acc = mt.accuracy_score(y_test, y_pred_svc)

svc_pol = SVC(kernel="poly")
svc_pol.fit(x_train,y_train)
y_pred_svc_pol = svc_pol.predict(x_test)

cross_svc_pol = cross_val_score(estimator = svc_pol, X=x_train, y = y_train, cv = 4)
y_pred2_svc_acc = cross_svc_pol.mean()

dog_değ_svc_pol = mt.confusion_matrix(y_test, y_pred_svc_pol)
y_pred1_svc_pol_acc = mt.accuracy_score(y_test, y_pred_svc_pol)


svc_result = {"i": [], "y_pred_svc" : [] , "cross_svc": [], "y_pred2_svc_acc" : [], "dog_değ_svc" : [], "y_pred1_svc_acc" : []}

for i in range(1,6):
    
    svc_result["i"].append(i)
    
    svc = SVC(kernel="rbf")
    svc.fit(x_train,y_train)
    y_pred_svc = svc.predict(x_test)
    
    svc_result["y_pred_svc"].append(y_pred_svc) 
    
    cross_svc = cross_val_score(estimator = svc, X=x_train, y = y_train, cv = 4)
    svc_result["cross_svc"].append(cross_svc)
    
    y_pred2_svc_acc = cross_svc.mean()
    svc_result["y_pred2_svc_acc"].append(y_pred2_svc_acc)
    
    dog_değ_svc = mt.confusion_matrix(y_test, y_pred_svc)
    svc_result["dog_değ_svc"].append(dog_değ_svc)
    
    y_pred1_svc_acc = mt.accuracy_score(y_test, y_pred_svc)
    svc_result["y_pred1_svc_acc"].append(y_pred1_svc_acc)

#%% ANN

kr = Sequential()

kr.add(Dense(16,kernel_initializer ='uniform',activation="tanh",input_dim= x_train.shape[1]))
kr.add(Dense(8,kernel_initializer ="uniform",activation="relu"))
kr.add(Dense(8,kernel_initializer ="uniform",activation="tanh"))
kr.add(Dense(1,kernel_initializer ="uniform",activation="sigmoid"))
    
kr.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


hist = kr.fit(x_train,y_train,epochs=50, validation_data=(x_test, y_test))
y_pred= kr.predict(x_test)

y_pred=y_pred>0.5

plt.figure(figsize = (17,5))
plt.subplot(1, 2, 1)
plt.plot(hist.history["accuracy"], label = "Train")
plt.plot(hist.history["val_accuracy"], label = "Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.history["loss"], label = "Train")
plt.plot(hist.history["val_loss"], label = "Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()

#%% kmeans

kmeans = KMeans(n_clusters= 7, init = "k-means++")
kmeans.fit(x_train)

print (kmeans.cluster_centers_)

y_pred_kmeans = kmeans.fit_predict(x_test)


# plt.scatter(x_train[y_pred_kmeans==0,0], x_train[y_pred_kmeans == 0,1], s = 100, c = "red")
# plt.scatter(x_train[y_pred_kmeans==1,0], x_train[y_pred_kmeans == 1,1], s = 100, c = "green")
# plt.scatter(x_train[y_pred_kmeans==2,0], x_train[y_pred_kmeans == 2,1], s = 100, c = "yellow")
# plt.scatter(x_train[y_pred_kmeans==3,0], x_train[y_pred_kmeans == 3,1], s= 100, c = "blue")
# plt.scatter(x_train[y_pred_kmeans==4,0], x_train[y_pred_kmeans == 4,1], s= 100, c = "pink")
# plt.scatter(x_train[y_pred_kmeans==5,0], x_train[y_pred_kmeans == 5,1], s= 100, c = "black")
# plt.scatter(x_train[y_pred_kmeans==6,0], x_train[y_pred_kmeans == 6,1], s= 100, c = "orange")
# plt.title("Kmeans")
# plt.show()

#%% Hirarcical

ac = AgglomerativeClustering(n_clusters = 4, affinity = "euclidean", linkage = "ward")
y_pred_he = ac.fit_predict(x_train)
print(y_pred_he)

print (ac.cluster_centers_)

y_pred_kmeans = ac.fit_predict(x_test)






























