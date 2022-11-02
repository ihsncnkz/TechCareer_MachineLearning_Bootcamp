# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 09:30:53 2022

@author: ihsan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veri = pd.read_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Veriler/musteriler.csv")

X = veri.iloc[:,3:].values

#%% Kmeans

from sklearn.cluster import KMeans

kms = KMeans(n_clusters = 4, init = "k-means++")
kms.fit(X)
y_pred = kms.fit_predict(X)
print(y_pred)
print(kms.cluster_centers_)

plt.scatter(X[y_pred==0,0], X[y_pred == 0,1], s = 100, c = "red")
plt.scatter(X[y_pred==1,0], X[y_pred == 1,1], s = 100, c = "green")
plt.scatter(X[y_pred==2,0], X[y_pred == 2,1], s = 100, c = "yellow")
plt.scatter(X[y_pred==3,0], X[y_pred == 3,1], s = 100, c = "blue")
plt.title("Kmeans")
plt.show()


sonuclar = []
for i in range(1,11):
    kms2 = KMeans(n_clusters = i, init="k-means++")
    kms2.fit(X)
    sonuclar.append(kms2.inertia_)
    
plt.plot(range(1,11), sonuclar)


#%% Hierarchical

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")
y_pred_he = ac.fit_predict(X)
print(y_pred_he)

plt.scatter(X[y_pred_he==0,0], X[y_pred_he == 0,1], s = 100, c = "red")
plt.scatter(X[y_pred_he==1,0], X[y_pred_he == 1,1], s = 100, c = "green")
plt.scatter(X[y_pred_he==2,0], X[y_pred_he == 2,1], s = 100, c = "yellow")
plt.title("Hierarchical")
plt.show()

ac2 = AgglomerativeClustering(n_clusters = 4, affinity = "euclidean",linkage = "ward")
y_pred_he2 = ac2.fit_predict(X)
print(y_pred_he2)

plt.scatter(X[y_pred_he2==0,0], X[y_pred_he2 == 0,1], s = 100, c = "red")
plt.scatter(X[y_pred_he2==1,0], X[y_pred_he2 == 1,1], s = 100, c = "green")
plt.scatter(X[y_pred_he2==2,0], X[y_pred_he2 == 2,1], s = 100, c = "yellow")
plt.scatter(X[y_pred_he2==3,0], X[y_pred_he2 == 3,1], s = 100, c = "blue")
plt.title("Hierarchical")
plt.show()


import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.show()






































