# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 10:48:35 2022

@author: ihsan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

veri = pd.read_csv("tenis.csv")

veriler2 = veri.apply(pr.LabelEncoder().fit_transform)

havadur = veriler2.iloc[:,:1]
havadurson = pr.OneHotEncoder().fit_transform(havadur).toarray()

havadursondf = pd.DataFrame(data = havadurson, index = range(14), columns = ["o","r","s"])

sonveri = pd.concat([havadursondf, veriler2.iloc[:,1:3]], axis = 1)
sonveri2 = pd.concat([sonveri.iloc[:,:3], veri.iloc[:,1:3]], axis = 1)

sonveri3 = pd.concat([sonveri2, veriler2.iloc[:,-2]], axis = 1)

Y = veriler2.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(sonveri3, Y, test_size = 0.33, random_state= 0)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

X = np.append(arr = np.ones((14,1)).astype(int), values = sonveri3 , axis = 1)

X_l = sonveri3.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(Y, X_l).fit()
print(r_ols.summary())

X_l2 = sonveri3.iloc[:,[0,1,2,4,5]].values
r_ols2 = sm.OLS(Y, X_l2).fit()
print(r_ols2.summary())


x_train, x_test, y_train, y_test = train_test_split(sonveri3.iloc[:,[0,1,2,4,5]], Y, test_size = 0.33, random_state= 0)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred2 = lr.predict(x_test)