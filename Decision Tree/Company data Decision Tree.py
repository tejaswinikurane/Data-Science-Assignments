# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:25:47 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('E:\Tej\Assignments\Asgnmnt\Decision Tree\Company_Data.csv')
data
data.describe()
data.info() #no null values
data.columns

data_cat = ['ShelveLoc','Urban', 'US']

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
for i in data_cat:
    data[i] = number.fit_transform(data[i])
    
label_enc = LabelEncoder()
data['sales'] = label_enc.fit_transform(pd.cut(data['Sales'],bins = 2, retbins = True )[0]) #converting data into two bins
data = data.drop('Sales',axis = 1) #dropping the existing Sales column

#EDA
import seaborn as sns
sns.countplot(x= 'ShelveLoc',data =data)
sns.countplot(x= 'Urban',data =data)
sns.countplot(x= 'US',data =data)

sns.barplot(data.sales,data.Price)
sns.barplot(data.sales,data.Income)
sns.barplot(data.Urban,data.Income)

sns.boxplot(x='CompPrice',data = data)
sns.boxplot(x ='Population',data = data) #no outlier
sns.boxplot(x='Price',data = data) 
sns.boxplot(x='Age',data =data) #no outlier
sns.boxplot(x= 'Education',data = data) #no outlier
sns.boxplot(x= 'Income',data = data) #no outlier
help(sns.boxplot)  

sns.boxplot(x = 'sales', y = 'CompPrice', data = data)
sns.boxplot(x = 'sales', y = 'Population', data = data)
sns.boxplot(x = 'sales', y = 'Price', data = data)
sns.boxplot(x = 'sales', y = 'Income', data = data)

data.sales.value_counts() #0=241,1=159
#imbalance in the sales variable will cause bias in the model. so, now we will go for resampling

majority_class = data[data.sales == 0] 
minority_class = data[data.sales == 1]

from sklearn.utils import resample
minority_class_unsampled = resample(minority_class,
                                    replace = True, #sample with replacement
                                    n_samples = 241, #samples matching majority class
                                    random_state = 123) #reproducing results

data_sampled = pd.concat([majority_class,minority_class_unsampled])
#determining x and y data
X= data_sampled.iloc[:,0:10]
Y= data_sampled.iloc[:,-1]

#separating train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.20,random_state = 42)

# building decision tree classification model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
model.fit(x_train,y_train)

#predicting on test data
y_pred = model.predict(x_test)
#getting accuracy score

#getting accuracy score
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred) #89.69%


pd.crosstab(y_test,y_pred)
model.score(x_train,y_train)

#accuracy score of test data
model.score(x_test,y_test) #0.896

#storing predictions to the data
data['y_pred'] = model.predict(data.iloc[:,0:10]) 
pd.crosstab(data.sales,data.y_pred).plot(kind = 'bar')

from sklearn.metrics import confusion_matrix
cm = print(confusion_matrix(data.sales, data.y_pred))
np.mean(data.y_pred == data.sales) #0.945

#overall accuracy
model.score(X,Y) #0.979
model.predict([[135,80,12,302,100,1,50,20,1,0]])

from sklearn import tree
tree.plot_tree(model.fit(x_train,y_train))


