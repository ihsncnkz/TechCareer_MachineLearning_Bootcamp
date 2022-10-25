# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:01:33 2022

@author: ihsan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

veri = pd.read_excel("E:/Work/Bootcamp/TechCareer/Machine_Learning/Veriler/sondata.xlsx")

