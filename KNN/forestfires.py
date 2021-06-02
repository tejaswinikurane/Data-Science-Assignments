# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:44:50 2021

@author: Admin
"""


# KNN Classification
from pandas import read_csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

filename = read_csv('E:\\Tej\\Assignments\\Asgnmnt\\KNN\\forestfires.csv')
names = [
    'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain','Size_Categorie'
]

dataframe = read_csv('E:\\Tej\\Assignments\\Asgnmnt\\KNN\\forestfires.csv', names=names)
array = dataframe.values
X = array[:, 4:11]
Y = array[:, -1]

num_folds = 10
kfold = KFold(n_splits=10)

model = KNeighborsClassifier(n_neighbors=17)
results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())