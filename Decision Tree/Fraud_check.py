# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:04:15 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('E:\\Tej\\Assignments\\Asgnmnt\\Decision Tree\\Fraud_check.csv')
data
data.info() #no-null values
data.columns
df = data['Taxable.Income']
data['status'] = np.where(df<=30000,'Risky','Good`')
data_cat = ['Undergrad', 'Marital.Status','Urban','status']

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
for i in data_cat:
    data[i] = number.fit_transform(data[i])

#EDA
import seaborn as sns
sns.countplot(x='Undergrad',data=data).plot(kind = 'bar')
sns.countplot(x='Marital.Status',data=data).plot(kind = 'bar')
sns.countplot(x='Urban',data=data).plot(kind = 'bar')
sns.countplot(x='status',data=data).plot(kind = 'bar') #largely imbalanced

sns.boxplot(x= 'Marital.Status',y= 'City.Population',data = data)
sns.boxplot(x= 'status',y= 'City.Population',data = data)
sns.boxplot(x= 'status',y= 'Work.Experience',data = data)

pd.crosstab(data['Marital.Status'],data.status).plot(kind = 'bar')
#dropping the Taxable.Income Column as we've converted it into the categorical variable
data = data.drop('Taxable.Income',axis = 1)

##Oversampling to avoid any information loss and to deal with the bias
data.status.value_counts() # 0=476,1=124
majority_class = data[data.status == 0]
minority_class = data[data.status == 1]

from sklearn.utils import resample
minority_class_unsampled = resample(minority_class,
                                    replace = True,  #sample with replacement
                                    n_samples = 476, #to match majority class
                                    random_state = 123) #reproducible results

df_unsampled = pd.concat([majority_class,minority_class_unsampled])
df_unsampled.status.value_counts() # 0=476,1 =476
pd.crosstab(df_unsampled['Marital.Status'],df_unsampled.status).plot(kind = 'bar') #no bias

#separating x and y variables
X= df_unsampled.iloc[:,0:5]
Y= df_unsampled.iloc[:,-1]

#splitting into test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.3,random_state = 42)
    
#building classification model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy',min_samples_split = 2)
model.fit(x_train,y_train)
model.predict(x_test)

#accuracy on test data
model.score(x_test,y_test) #0.8461

#storing the predictions
data['y_pred'] = model.predict(data.iloc[:,0:5]) 
pd.crosstab(data.status,data.y_pred)

#getting accuracy scores
model.score(data.iloc[:,0:5],data.iloc[:,-1]) #1.0
model.score(X,Y) #0.9538

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(data.status,data.y_pred)
cm

#Visualizing the tree
from sklearn import tree
tree.plot_tree(model.fit(x_train,y_train))


