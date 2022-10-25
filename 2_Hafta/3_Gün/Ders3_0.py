# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 09:30:14 2022

@author: ihsan
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

veriler = pd.read_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Veriler/maaslar.csv")

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

K = x + 0.5

Z = x - 0.4

W = x - 0.5

rf_reg = RandomForestRegressor(random_state = 0, n_estimators = 15)
rf_reg.fit(x,y.ravel())
y_pred = rf_reg.predict(x)

plt.scatter(x, y, color = "red")
plt.plot(x, y_pred, color = "blue")
plt.plot(x,rf_reg.predict(Z), color = "green")
plt.plot(x,rf_reg.predict(K), color = "yellow")
plt.plot(x,rf_reg.predict(W), color = "pink")
plt.show()

print(rf_reg.predict([[11]]))
print(rf_reg.predict([[6.6]]))

# R^2
print("Random Forest R2 Score: ", r2_score(y,y_pred))
