# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:50:32 2022

@author: ihsan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

veri = pd.read_csv("maaslar.csv")

x = veri.iloc[:,1:2]
y = veri.iloc[:,2:]

X = x.values
Y = y.values

lr = LinearRegression()
lr.fit(X,Y)
y_pred = lr.predict(X)

plt.scatter(X, Y, color = "red")
plt.plot(X, y_pred, color = "blue")
plt.show()

pf = PolynomialFeatures()
lr2 = LinearRegression()
x_pol = pf.fit_transform(X)


lr2.fit(x_pol, y)
y_pred_pol = lr2.predict(pf.fit_transform(X))

plt.scatter(X, Y, color = "red")
plt.plot(X, y_pred_pol, color = "blue")
plt.show()

#%%

pf = PolynomialFeatures(degree = 3)
lr2 = LinearRegression()
x_pol2 = pf.fit_transform(X)

lr2.fit(x_pol2, y)
y_pred_pol = lr2.predict(pf.fit_transform(X))

plt.scatter(X, Y, color = "red")
plt.plot(X, y_pred_pol, color = "blue")
plt.show()

#%%

pf = PolynomialFeatures(degree = 4)
lr2 = LinearRegression()
x_pol3 = pf.fit_transform(X)

lr2.fit(x_pol3, y)
y_pred_pol3 = lr2.predict(pf.fit_transform(X))

plt.scatter(X, Y, color = "red")
plt.plot(X, y_pred_pol3, color = "blue")
plt.show()

