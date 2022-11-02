# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:20:12 2022

@author: ihsan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer

veri = pd.read_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Veriler/iris.csv")

X = veri.iloc[:,:-1].values

#%% Kmeans

Kmeans = KMeans(n_clusters=3, init = "k-means++")
Kmeans.fit(X)
print(Kmeans.cluster_centers_)

Kmeans_sonuc = []
for i in range(1,11):
    Kmeans2 = KMeans(n_clusters = i, init="k-means++")
    Kmeans2.fit(X)
    Kmeans_sonuc.append(Kmeans2.inertia_)
plt.plot(range(1,11), Kmeans_sonuc)
plt.show()


Kmeans3 = KMeans(n_clusters=4, init = "k-means++")
Kmeans3.fit(X)
y_pred_Kmeans3 = Kmeans3.fit_predict(X)
print(y_pred_Kmeans3)
print(Kmeans3.cluster_centers_)

plt.scatter(X[y_pred_Kmeans3==0,0], X[y_pred_Kmeans3 == 0,1], s = 100, c = "red")
plt.scatter(X[y_pred_Kmeans3==1,0], X[y_pred_Kmeans3 == 1,1], s = 100, c = "green")
plt.scatter(X[y_pred_Kmeans3==2,0], X[y_pred_Kmeans3 == 2,1], s = 100, c = "yellow")
plt.scatter(X[y_pred_Kmeans3==3,0], X[y_pred_Kmeans3 == 3,1], s= 100, c = "blue")
plt.title("Kmeans")
plt.show()


visu = KElbowVisualizer(Kmeans3, k = (2,12), metric = "distortion", init = "k-means++", timings= False, locate_elbow=True)
visu.fit(X)
visu.poof()

#%% Hierarchical

ac = AgglomerativeClustering(n_clusters = 4, affinity = "euclidean", linkage = "ward")
y_pred_he = ac.fit_predict(X)
print(y_pred_he)

plt.scatter(X[y_pred_he==0,0], X[y_pred_he == 0,1], s = 100, c = "red")
plt.scatter(X[y_pred_he==1,0], X[y_pred_he == 1,1], s = 100, c = "green")
plt.scatter(X[y_pred_he==2,0], X[y_pred_he == 2,1], s = 100, c = "yellow")
plt.scatter(X[y_pred_he==3,0], X[y_pred_he == 3,1], s = 100, c = "blue")
plt.title("Hierarchical")
plt.show()

ac2 = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean",linkage = "ward")
y_pred_he2 = ac2.fit_predict(X)
print(y_pred_he2)

plt.scatter(X[y_pred_he2==0,0], X[y_pred_he2 == 0,1], s = 100, c = "red")
plt.scatter(X[y_pred_he2==1,0], X[y_pred_he2 == 1,1], s = 100, c = "green")
plt.scatter(X[y_pred_he2==2,0], X[y_pred_he2 == 2,1], s = 100, c = "yellow")
plt.title("Hierarchical")
plt.show()

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.show()