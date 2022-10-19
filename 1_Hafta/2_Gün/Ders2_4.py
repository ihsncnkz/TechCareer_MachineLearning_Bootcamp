# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:20:34 2022

@author: ihsan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

veri = pd.read_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Ders2/veriler.csv")

x = veri.iloc[:,1:3]
y = veri.iloc[:,3:4]

X = x.values
Y = y.values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)

# Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_train)

"""
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, y_pred)
plt.show()
"""

# Polinomial Linear Regression
pf = PolynomialFeatures()
lr2 = LinearRegression()
x_pol = pf.fit_transform(X)
x_train_poly, x_test_poly, y_train_poly, y_test_poly = train_test_split(x_pol, Y, test_size = 0.33, random_state = 0)

lr2.fit(x_train_poly, y_train_poly)
y_pred_pol0 = lr2.predict(x_train_poly)

# StandartScalar
sc = StandardScaler()
x_train_olc = sc.fit_transform(x_train)
x_test_olc = sc.fit_transform(x_test)
y_train_olc = sc.fit_transform(y_train)
y_test_olc = sc.fit_transform(y_test)

# SVR

svr = SVR(kernel = "rbf")

svr.fit(x_train_olc, y_train_olc)

y_pred_svr0 = svr.predict(x_train_olc)


# DTR

dtr = DecisionTreeRegressor(random_state = 0)

dtr.fit(x_train,y_train)

y_pred_dtr0 = dtr.predict(x_train)

































