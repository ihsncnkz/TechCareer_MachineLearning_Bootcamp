# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:08:10 2022

@author: ihsan
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv("satislar.csv")

aylar = np.array(data[["Aylar"]]).reshape(-1,1)
satislar = np.array(data[["Satislar"]]).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size = 0.33, random_state= 0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

lr = LinearRegression()
lr.fit(aylar, satislar)
y_pred = lr.predict(aylar)

lr.fit(aylar, satislar)
y_pred2 = lr.predict(aylar)

plt.scatter(aylar, satislar, color = "red")
plt.plot(x_test, y_pred2, color = "blue")
plt.show()