# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:04:36 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv('E:\\Tej\\Assignments\\Asgnmnt\\KNN\\glass.csv')
data

data.describe()
data.info()
data.nunique()
data.columns

#Preprocessing 
#standardization
def norm_func(i):
    x= (i - i.min())/(i.max()-i.min())
    return(x)

data.iloc[:,[0,1,2,3,4,6,]] = norm_func(data.iloc[:,[0,1,2,3,4,6,]])

#EDA 
cor = data.corr()
sns.heatmap(cor)
# =============================================================================
# help(np.corrcoef)
# Ca and Ri are highly correlated, therefore we will Ca or Ri(in this case Ca)
# Ca and K have very low correlation, 
# K is having very less correlation with type, herefore we will drop K also
# =============================================================================
g= sns.pairplot(data,hue = 'Type',diag_kind = 'hist')

data = data.drop(data.iloc[:,[5,6]],axis =1)
#data are highly uncorrelated

sns.boxplot(x= 'Type',y= 'RI',data =data)
sns.boxplot(x= 'Type',y= 'Na',data =data)
sns.boxplot(x= 'Type',y= 'Mg',data =data)
sns.boxplot(x= 'Type',y= 'Al',data =data)
sns.boxplot(x= 'Type',y= 'Si',data =data)
sns.boxplot(x= 'Type',y= 'K',data =data)
sns.boxplot(x= 'Type',y= 'Ca',data =data)

# Target and independent variables
X = data.iloc[:,0:7]
Y = data.iloc[:,-1]

#splitting data into train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.25,random_state = 42)

####conventional method####
#KNN classification
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(x_train,y_train)
model.predict(x_test)

#training accuracy
model.score(x_train,y_train) #0.775
#test accuracy
model.score(x_test,y_test)   #0.703

#storing the results for original dataset
data['y_pred'] = model.predict(X)

#overall mean accuracy
np.mean(data.Type == data.y_pred) #0.757


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
#highest accuracy is obtained with 3 number of clusters that is 75.92%