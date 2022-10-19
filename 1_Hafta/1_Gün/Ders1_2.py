# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 13:47:28 2022

@author: ihsan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("veriler.csv")

x = data.iloc[:,1:4].values
y = data.iloc[:,4:].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state= 0 )

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

log = LogisticRegression(random_state = 0)

log.fit(X_train, y_train)

y_pred = log.predict(X_test)

