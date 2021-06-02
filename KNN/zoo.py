# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 20:26:39 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('E:\\Tej\\Assignments\\Asgnmnt\\KNN\\Zoo.csv')
data

#EDA and Preprocessing
data.describe()
data.info() #no null values
data.nunique()
data.columns

for col in data:
    plt.figure(figsize = (15,10))
    sns.barplot(data[col].value_counts().index,data[col].value_counts())
    plt.show()
# =============================================================================
# most animals don't have feathers
# less number of animals are airborne
# most animals do not have backbone
# not animals animals are mostly categorized into 4 and 7 groups
# very less number of animals are venomous
# very less number of animals have fins
# mostly animals are not domestic
# most animals have 4 legs
# =============================================================================

X = data.iloc[:,1:17]
Y = data.iloc[:,-1]

#splitting into train test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.25, random_state = 42)

####conventional method####
#KNN classification
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(x_train,y_train)
model.predict(x_test)

#train accuracy
model.score(x_train,y_train) #0.9333

#test accuracy
model.score(x_test,y_test) #0.8846

#predicting esults on original dataset
data['y_pred'] = model.predict(X)
#overall accuray
model.score(X,Y) #0.92

#getting dataframe of outcomes comparing with original types
outcome = data[['animal name','type','y_pred']].sort_values('y_pred')

####Using for Loop####
from sklearn import metrics
k_range  = range(1,10)
scores={}
scores_list = []
for k in k_range:
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    scores[k]= metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))
    
print(scores_list)
#highest accuracy is obtained with 2 and 3 number of clusters that is 100% and 96.15%
