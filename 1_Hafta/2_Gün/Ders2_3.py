# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 13:49:45 2022

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

veri = pd.read_csv("maaslar.csv")

x = veri.iloc[:,1:2]
y = veri.iloc[:,2:]

X = x.values
Y = y.values

# Linear Regression
lr = LinearRegression()
lr.fit(X,Y)
y_pred = lr.predict(X)

plt.scatter(X, Y, color = "red")
plt.plot(X, y_pred, color = "blue")
plt.show()

# Polinomial Linear Regression
pf = PolynomialFeatures(degree = 4)
lr2 = LinearRegression()
x_pol = pf.fit_transform(X)

lr2.fit(x_pol, y)
y_pred_pol = lr2.predict(x_pol)

plt.scatter(X, Y, color = "red")
plt.plot(X, y_pred_pol, color = "blue")
plt.show()

#%% SVR

# StandardScaler = Normalizasyon
sc = StandardScaler()
X_olc = sc.fit_transform(X)
sc2 = StandardScaler()
y_olc = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

# SVR

svr = SVR(kernel = "rbf")

svr.fit(X_olc, y_olc)

y_pred_svr0 = svr.predict(X_olc)

plt.scatter(X_olc, y_olc, color = "red")
plt.plot(X_olc, y_pred_svr0)
plt.show()

# SVR Polly

svr2 = SVR(kernel = "poly", degree = 3)

svr2.fit(X_olc, y_olc)

y_pred_svr1 = svr2.predict(X_olc)

plt.scatter(X_olc, y_olc, color = "red")
plt.plot(X_olc, y_pred_svr1)
plt.show()

# SVR Polly degree 4

svr3 = SVR(kernel = "poly", degree = 4)

svr3.fit(X_olc, y_olc)

y_pred_svr2 = svr3.predict(X_olc)

plt.scatter(X_olc, y_olc, color = "red")
plt.plot(X_olc, y_pred_svr2)
plt.show()

# SVR polly 

svr4 = SVR(kernel = "rbf")

svr.fit(X, Y)

y_pred_svr4 = svr.predict(X)

plt.scatter(X_olc, y_olc, color = "red")
plt.plot(X_olc, y_pred_svr0)
plt.show()

#%% Decesion Tree

dtr = DecisionTreeRegressor(random_state = 0)

dtr.fit(X,Y)

y_pred_dtr0 = dtr.predict(X)

Z = X + 0.5

W = X - 0.56

K = X - 0.4

plt.scatter(X, Y, color = "red")
plt.plot(X, y_pred_dtr0, color = "blue")

plt.plot(X, dtr.predict(Z), color = "yellow")
plt.plot(X, dtr.predict(K), color = "gray")
plt.plot(X, dtr.predict(W), color = "green")

plt.show()

print(dtr.predict([[11]]))
print(dtr.predict([[6.6]]))
print(dtr.predict([[20]]))




















