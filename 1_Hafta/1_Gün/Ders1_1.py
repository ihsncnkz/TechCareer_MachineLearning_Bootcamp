# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 11:43:52 2022

@author: ihsan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler

eksik_veriler = pd.read_csv("eksikveriler.csv")

# missin valuları np kütphanesin alıcaksın ve o sütünün ortalama değerlerini yazacaksın.
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean") 

yas = eksik_veriler.iloc[:,1:4].values

imputer = imputer.fit(yas[:,1:4])

yas[:,1:4] = imputer.transform(yas[:,1:4])

y = eksik_veriler.iloc[:,-1].values
x = yas

x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.33, random_state= 0 )

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

