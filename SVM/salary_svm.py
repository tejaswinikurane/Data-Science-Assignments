# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:09:02 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('E:\\Tej\\Assignments\\Asgnmnt\\SVM\\SalaryData_Train.csv')
df_test  = pd.read_csv('E:\\Tej\\Assignments\\Asgnmnt\\SVM\\SalaryData_Test.csv')

df_train.describe()
df_train.info()
df_train.columns

df_cat = ['workclass', 'education','maritalstatus','occupation', 'relationship', 'race', 'sex','native', 'Salary']

from sklearn.preprocessing import LabelEncoder
number =LabelEncoder()

for i in df_cat:
    df_train[i] = number.fit_transform(df_train[i])
    
for i in df_cat:
    df_test[i] = number.fit_transform(df_test[i])
    
x_train = df_train.iloc[:,0:13]
y_train = df_train.iloc[:,-1]
x_test = df_test.iloc[:,0:13]
y_test = df_test.iloc[:,-1]

#barplot of the data
import seaborn as sns
for col in df_train:
    plt.figure(figsize=(10,4))
    sns.barplot(df_train[col].value_counts().index,df_train[col].value_counts())
    plt.tight_layout()
    
##rbf kernel
from sklearn.svm import SVC
model = SVC(kernel = 'rbf')
model.fit(x_train,y_train)

model.predict(x_test)
model.score(x_test,y_test) #0.7964

##poly kernel
model_poly = SVC(kernel = 'poly')
model_poly.fit(x_train,y_train)

model_poly.predict(x_test)
model_poly.score(x_test,y_test) #0.7795

##sigmoid kernel.
model_sig = SVC(kernel = 'sigmoid')
model_sig.fit(x_train,y_train)

model.predict(x_test)
model_sig.score(x_test,y_test) #0.7568

#we will go for rbf kernel giving maximum accuracy
