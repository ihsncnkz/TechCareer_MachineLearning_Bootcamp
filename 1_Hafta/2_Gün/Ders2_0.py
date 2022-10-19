# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 10:28:37 2022

@author: ihsan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.linear_model import LinearRegression
from Startmodel.api import OLS

data = pd.read_csv("veriler.csv")

ybk = data.iloc[:,1:4].values

ulke = data.iloc[:,0:1].values

cinsiyet = data.iloc[:,-1:].values

le = preprocessing.LabelEncoder()

Ohe = preprocessing.OneHotEncoder()

ulkeLE = le.fit_transform(ulke)

ulkeLEDF = pd.DataFrame(data = ulkeLE, index = range(22), columns = ["ulke"])

ulkeson = Ohe.fit_transform(ulke).toarray()

cinsiyet[:,-1] = le.fit_transform(cinsiyet[:,-1])

ulkedf = pd.DataFrame(data = ulkeson, index = range(22), columns= ["fr", "tr",  "us"] )

cinsiyetdf = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"])

ybkdf = pd.DataFrame(data = ybk, index = range(22), columns=["boy","kilo","yas"])

ulkeybk = pd.concat([ulkeLEDF, ybkdf], axis = 1)

verilerson = pd.concat([ulkedf, ybkdf, cinsiyetdf], axis = 1)


X = data.iloc[:,:-1].values
Y = data.iloc[:,-1:].values

X = np.append(arr = np.ones((22,1)).astype(int), values = ulkeybk, axis = 1)
x_l = np.array(X_l, dtype= float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())