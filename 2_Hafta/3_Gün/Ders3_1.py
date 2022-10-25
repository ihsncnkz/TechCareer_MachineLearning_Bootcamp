# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:24:51 2022

@author: ihsan
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

veri = pd.read_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Veriler/maaslar_yeni.csv")

x = veri.iloc[:,2:5].values
y = veri.iloc[:,5:].values

unvan = veri.iloc[:,2:3]
puan = veri.iloc[:,4:5]

# x_new = veri.iloc[:,2:5:2]

x_yeni = pd.concat([unvan, puan], axis = 1)
x_yeni = x_yeni.values

lr = LinearRegression()
lr.fit(x_yeni,y)
y_lr_pred = lr.predict(x_yeni)

model = sm.OLS(lr.predict(x_yeni),x_yeni)
print(model.fit().summary())

print("R^2 Score: ",r2_score(y, y_lr_pred))

print("-------------------------------------------")
#%% Polinom

pol_reg = PolynomialFeatures(degree = 4)
x_poly = pol_reg.fit_transform(x_yeni)

lr_pol = LinearRegression()
lr_pol.fit(x_poly, y)
y_pol_pred = lr_pol.predict(x_poly)

model_pol = sm.OLS(y_pol_pred,x_yeni)
print(model_pol.fit().summary())

print("R^2 Score: ", r2_score(y, y_pol_pred))

print("-------------------------------------------")
#%% Standarscaler
sc = StandardScaler()
x_sc = sc.fit_transform(x_yeni)
y_sc = sc.fit_transform(y)

#%% SVR

svr = SVR(kernel = "rbf")

svr.fit(x_sc, y_sc)

y_SVR_pred = svr.predict(x_sc)

model_svr = sm.OLS(y_SVR_pred,x_sc)
print(model_svr.fit().summary())

print("R^2 Score: ", r2_score(y_sc, y_SVR_pred))

print("-------------------------------------------")
#%% DTR

dtr = DecisionTreeRegressor(random_state = 0)

dtr.fit(x_yeni,y)

y_dtr_pred = dtr.predict(x_yeni)

model_dtr = sm.OLS(y_dtr_pred,x_yeni)
print(model_svr.fit().summary())

print("R^2 Score: ", r2_score(y, y_dtr_pred))

print("-------------------------------------------")
#%% RFR

rf_reg = RandomForestRegressor(random_state = 0, n_estimators = 15)
rf_reg.fit(x_yeni,y.ravel())
y_rfr_pred = rf_reg.predict(x_yeni)

model_rfr = sm.OLS(y_rfr_pred,x_yeni)
print(model_svr.fit().summary())

print("R^2 Score: ", r2_score(y, y_rfr_pred))





















