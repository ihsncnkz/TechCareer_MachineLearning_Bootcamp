# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 10:25:00 2022

@author: ihsan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.read_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Ders1/veriler.csv")

boykilo = data[["boy","kilo"]]
boykilo2 = data.iloc[:,1:3].values

ulke = data.iloc[:,0:1].values

le = LabelEncoder()
ulke[:,0] = le.fit_transform(data.iloc[:,0])

ohe = OneHotEncoder() # veri seti içerisinde ulke değerlerinin olup olmadığına bakıyoruz var : 1 yok: 0
ulke = ohe.fit_transform(ulke).toarray()

ulkeson = pd.DataFrame(data = ulke, index = range(22), columns= ["fr", "tr",  "us"])


# Csv formatında birleştirme.
# ulkeson.to_csv("E:/Work/Bootcamp/TechCareer/Machine_Learning/Ders1/veriler.csv")

yasboykilo = data[["boy", "kilo", "yas"]]
ybk = data.iloc[:,1:4].values

ybkDF = pd.DataFrame(data = ybk, columns=["boy","kilo","yas"])

cinsiyet = data.iloc[:,4]
cinsiyetdf = pd.DataFrame(data = cinsiyet, columns = ["cinsiyet"])

ulkeybk = pd.concat([ulkeson, ybkDF], axis = 1)

verilerson = pd.concat([ulkeybk, cinsiyetdf], axis = 1)

"""
ödev veri seti bulun kendinize ve bu adımları o veri setine uygulayın. Bulduğunuz veri setini
projenizde kullanacağınız veri seti olsa iyi olur. çünkü bize verilen projeyi yapmaya başlarız.
"""

