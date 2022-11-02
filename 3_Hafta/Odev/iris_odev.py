# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:31:34 2022

@author: ihsan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection as ms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import activations
#%% Veri Seti Hazırlama bolumu

veri =pd.read_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Veriler/iris.csv")

X = veri.iloc[:,:-1].values
Y = veri.iloc[:,-1].values

# Veri setine StandardScaler uygulanması
sc = StandardScaler()
X = sc.fit_transform(X)
y = preprocessing.LabelEncoder().fit_transform(Y)

# traiin test ayırıö
x_train, x_test, y_train, y_test= ms.train_test_split(X,y,test_size=0.33,random_state=0)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(x_train)
X_test_pca = pca.fit_transform(x_test)

lda=LDA(n_components=2)
X_train_lda = lda.fit_transform(x_train,y_train)
X_test_lda = lda.transform(x_test)

#%% SVC

print("PCA ile SVC Model")
svc_pca = SVC()

p_svc = [{'C':[1,2,3,4,5],'kernel':['linear']},
    {'C':[1,2,3,4,5],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001]},
    {'C':[1,2,3,4,5],'kernel':['poly'],'degree':[1,2,3,4,5,6,7],'gamma':[1,0.5,0.1,0.01,0.001]}]

grid = GridSearchCV(estimator = svc_pca, param_grid = p_svc, scoring = "accuracy", cv = 4)
grid_search = grid.fit(X_train_pca, y_train)
y_pred_svc_pca = grid_search.predict(X_test_pca)

best_parm_grid_pca = grid_search.best_params_
best_score_grid_pca = grid_search.best_score_

print("Best parameters of SVC model with PCA and gridseach function: ", best_parm_grid_pca)
print("Best score of SVC model PCA and gridsearch function: ", best_score_grid_pca )

print("---------------------------------")

print("LDA ile SVC Model")

svc_lda = SVC()

p_svc_lda = [{'C':[1,2,3,4,5],'kernel':['linear']},
    {'C':[1,2,3,4,5],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001]},
    {'C':[1,2,3,4,5],'kernel':['poly'],'degree':[1,2,3,4,5],'gamma':[1,0.5,0.1,0.01,0.001]}]

grid = GridSearchCV(estimator = svc_lda, param_grid = p_svc_lda, scoring = "accuracy", cv = 4)
grid_search = grid.fit(X_train_lda, y_train)
y_pred_svc_lda = grid_search.predict(X_test_lda)

best_parm_grid_lda = grid_search.best_params_
best_score_grid_lda = grid_search.best_score_

print("Best parameters of SVC model with LDA, gridseach function: ", best_parm_grid_lda)
print("Best score of SVC model LDA, gridsearch function: ", best_score_grid_lda )

print("---------------------------------")

#%% LogisticRegression

print("PCA ile Logistic Regression")

logr_pca = LogisticRegression(random_state = 0)
logr_pca.fit(X_train_pca, y_train)
y_pred_pca = logr_pca.predict(X_test_pca)

cm_pca = confusion_matrix(y_test, y_pred_pca)
print(cm_pca)

bas_log_pca = cross_val_score(estimator = logr_pca, X = X_train_pca, y = y_train, cv = 4)
print("LogisticRegression PCA Score: ", logr_pca.score(X_test_pca, y_test))
print(bas_log_pca.mean())
print(bas_log_pca.std())

print("---------------------------------")

print("LDA ile Logistic Regression")

logr_lda = LogisticRegression(random_state = 0)
logr_lda.fit(X_train_lda, y_train)
y_pred_lda = logr_lda.predict(X_test_lda)

cm_lda = confusion_matrix(y_test, y_pred_lda)
print(cm_lda)

bas_log_lda = cross_val_score(estimator = logr_lda, X = X_train_lda, y = y_train, cv = 4)
print("LogisticRegression LDA Score: ", logr_lda.score(X_test_lda, y_test))
print(bas_log_lda.mean())
print(bas_log_lda.std())
print("---------------------------------")

#%% KNN

print("PCA ile KNN, GridSearch")

knn_grid_pca = KNeighborsClassifier(metric = "minkowski")

p_knn = {"n_neighbors" : range(1,10), "weights" : ["uniform", "distance"], "p" : [1,2]}

grid_knn_pca = GridSearchCV(estimator = knn_grid_pca, param_grid = p_knn, scoring = "accuracy", cv = 4)

grid_knn_search_pca = grid_knn_pca.fit(X_train_pca, y_train)
y_pred_grid_knn_pca = grid_knn_pca.predict(X_test_pca)

best_parm_grid_knn_pca = grid_knn_search_pca.best_params_
best_score_grid_knn_pca = grid_knn_search_pca.best_score_

print("Best prameter of KNN with PCA, gridseach function: ", best_parm_grid_knn_pca)
print("Best score of KNN with PCA, gridsearch function: ",best_score_grid_knn_pca)

print("---------------------------------")

print("LDA ile KNN, GridSearch")

knn_grid_lda = KNeighborsClassifier(metric = "minkowski")

grid_knn_lda = GridSearchCV(estimator = knn_grid_lda, param_grid = p_knn, scoring = "accuracy", cv = 4)

grid_knn_search_lda = grid_knn_lda.fit(X_train_lda, y_train)
y_pred_grid_knn_lda = grid_knn_lda.predict(X_test_lda)

best_parm_grid_knn_lda = grid_knn_search_lda.best_params_
best_score_grid_knn_lda = grid_knn_search_lda.best_score_

print("Best prameter of KNN with LDA, gridseach function: ", best_parm_grid_knn_lda)
print("Best score of KNN with LDA, gridsearch function: ",best_score_grid_knn_lda)

print("---------------------------------")
#%% naive baise

print("PCA ilde GaussianNB")

gnb_pca = GaussianNB()
gnb_pca.fit(X_train_pca, y_train)
y_gnb_pred_pca = gnb_pca.predict(X_test_pca)

cm_gnb_pca = confusion_matrix(y_test, y_gnb_pred_pca)
print(cm_gnb_pca)

bas_svc_gnb_pca = cross_val_score(estimator = gnb_pca, X = X_train_pca, y = y_train,cv = 4)
print("GaussianNB PCA Score: ", gnb_pca.score(X_test_pca, y_test))
print(bas_svc_gnb_pca.mean())
print(bas_svc_gnb_pca.std())

print("---------------------------------")

print("LDA ilde GaussianNB")

gnb_lda = GaussianNB()
gnb_lda.fit(X_train_lda, y_train)
y_gnb_pred_lda = gnb_lda.predict(X_test_lda)

cm_gnb_lda = confusion_matrix(y_test, y_gnb_pred_lda)
print(cm_gnb_lda)

bas_svc_gnb_lda = cross_val_score(estimator = gnb_lda, X = X_train_lda, y = y_train,cv = 4)
print("GaussianNB LDA Score: ", gnb_lda.score(X_test_lda, y_test))
print(bas_svc_gnb_lda.mean())
print(bas_svc_gnb_lda.std())

print("---------------------------------")

print("PCA ile BernoulliNB")

bnb_pca = BernoulliNB()
bnb_pca.fit(X_train_pca,y_train)
y_pred_bnb_pca = bnb_pca.predict(X_test_pca)

cm_bnb_pca = confusion_matrix(y_test, y_pred_bnb_pca)
print(cm_bnb_pca)

print("PCA Sınıflandırma Raporu \n",metrics.classification_report(y_test, y_pred_bnb_pca))
print("PCA Doğruluk Değeri :", metrics.accuracy_score(y_test, y_pred_bnb_pca))

bas_na_pca = cross_val_score(estimator = bnb_pca, X = X_train_pca, y = y_train, cv = 5)
print("BernoulliNB PCA  Score: ", bnb_pca.score(X_test_pca, y_test))
print(bas_na_pca.mean())
print(bas_na_pca.std())

print("---------------------------------")

print("LDA ile BernoulliNB")

bnb_lda = BernoulliNB()
bnb_lda.fit(X_train_lda,y_train)
y_pred_bnb_lda = bnb_lda.predict(X_test_lda)

cm_bnb_lda = confusion_matrix(y_test, y_pred_bnb_lda)
print(cm_bnb_lda)

print("LDA Sınıflandırma Raporu \n",metrics.classification_report(y_test, y_pred_bnb_lda))
print("LDA Doğruluk Değeri :", metrics.accuracy_score(y_test, y_pred_bnb_lda))

bas_na_lda = cross_val_score(estimator = bnb_lda, X = X_train_lda, y = y_train, cv = 5)
print("BernoulliNB LDA Score: ", bnb_pca.score(X_test_lda, y_test))
print(bas_na_lda.mean())
print(bas_na_lda.std())

print("---------------------------------")

#%% decision tree

print("PCA ile DecisionTreeClassifier")

dtr_pca = DecisionTreeClassifier()

p_dtc = {"criterion" : ["gini", "entropy"], "splitter" : ["best", "random"]}

grid_dtc_pca = GridSearchCV(estimator = dtr_pca, param_grid = p_dtc, scoring = "accuracy", cv = 4)
grid_dtc_search_pca = grid_dtc_pca.fit(X_train_pca, y_train)
y_pred_grid_dtc_pca = grid_dtc_pca.predict(X_test_pca)

best_parm_grid_dtc_pca = grid_dtc_search_pca.best_params_
best_score_grid_dtc_pca = grid_dtc_search_pca.best_score_

print("Best prameter of DecisionTreeClassifier with PCA, gridseach function: ", best_parm_grid_dtc_pca)
print("Best score of DecisionTreeClassifier with PCA, gridsearch function: ",best_score_grid_dtc_pca)

print("---------------------------------")

print("LDA ile DecisionTreeClassifier")

dtr_lda = DecisionTreeClassifier()

grid_dtc_lda = GridSearchCV(estimator = dtr_lda, param_grid = p_dtc, scoring = "accuracy", cv = 4)
grid_dtc_search_lda = grid_dtc_lda.fit(X_train_lda, y_train)
y_pred_grid_dtc_lda = grid_dtc_lda.predict(X_test_lda)

best_parm_grid_dtc_lda = grid_dtc_search_lda.best_params_
best_score_grid_dtc_lda = grid_dtc_search_lda.best_score_

print("Best prameter of DecisionTreeClassifier with LDA, gridseach function: ", best_parm_grid_dtc_lda)
print("Best score of DecisionTreeClassifier with LDA, gridsearch function: ",best_score_grid_dtc_lda)

print("---------------------------------")

#%% Random Forest

print("PCA RandomForestClassifier")

rfc_pca = RandomForestClassifier()

p_rfc = {"n_estimators" : range(1,50), "criterion" : ["gini", "entropy"], "class_weight" : ["balanced", "balanced_subsample"]}

grid_rfc_pca = GridSearchCV(estimator = rfc_pca, param_grid = p_rfc, scoring = "accuracy", cv = 4)
grid_rfc_search_pca = grid_rfc_pca.fit(X_train_pca, y_train)
y_pred_grid_rfc_pca = grid_rfc_pca.predict(X_test_pca)
    
best_parm_grid_rfc_pca = grid_rfc_search_pca.best_params_
best_score_grid_rfc_pca = grid_rfc_search_pca.best_score_

print("Best prameter of RandomForestClassifier with PCA, gridseach function: ", best_parm_grid_rfc_pca)
print("Best score of RandomForestClassifier with PCA, gridsearch function: ",best_score_grid_rfc_pca)
print("---------------------------------")

print("LDA RandomForestClassifier")

rfc_lda = RandomForestClassifier()

grid_rfc_lda = GridSearchCV(estimator = rfc_lda, param_grid = p_rfc, scoring = "accuracy", cv = 4)
grid_rfc_search_lda = grid_rfc_lda.fit(X_train_lda, y_train)
y_pred_grid_rfc_lda = grid_rfc_lda.predict(X_test_lda)
    
best_parm_grid_rfc_lda = grid_rfc_search_pca.best_params_
best_score_grid_rfc_lda = grid_rfc_search_pca.best_score_

print("Best prameter of RandomForestClassifier with LDA, gridseach function: ", best_parm_grid_rfc_lda)
print("Best score of RandomForestClassifier with LDA, gridsearch function: ",best_score_grid_rfc_lda)
print("---------------------------------")

#%% kmeans

kmeans = KMeans(n_clusters= 4, init = "k-means++")
kmeans.fit(X)

print (kmeans.cluster_centers_)

y_pred_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_pred_kmeans==0,0], X[y_pred_kmeans == 0,1], s = 100, c = "red")
plt.scatter(X[y_pred_kmeans==1,0], X[y_pred_kmeans == 1,1], s = 100, c = "green")
plt.scatter(X[y_pred_kmeans==2,0], X[y_pred_kmeans == 2,1], s = 100, c = "yellow")
plt.scatter(X[y_pred_kmeans==3,0], X[y_pred_kmeans == 3,1], s= 100, c = "blue")
plt.title("Kmeans")
plt.show()

#%% Hirarcical

ac = AgglomerativeClustering(n_clusters = 4, affinity = "euclidean", linkage = "ward")
y_pred_he = ac.fit_predict(X)
print(y_pred_he)

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


























