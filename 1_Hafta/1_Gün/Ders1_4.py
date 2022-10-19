# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:51:45 2022

@author: ihsan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.linear_model import LinearRegression

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

x_train, x_test, y_train, y_test = train_test_split(ulkeybk, cinsiyetdf, test_size = 0.33, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

lr.fit(x_train, y_train)
y_pred2 = lr.predict(x_test)






























