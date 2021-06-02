# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:38:22 2020

@author: Admin
"""


import pandas as pd
import numpy as np

data = pd.read_csv('E:\\Tej\\Assignments\\Asgnmnt\\Random forest\\Company_Data.csv')
data.info()
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


#Building Random forest classifier
#After each tree is built, all of the data are run down the tree, and proximities are computed for
# each pair of cases. If two cases occupy the same terminal node, their proximity is increased by one.
# At the end of the run, the proximities are normalized by dividing by the number of trees. Proximities
# are used in replacing missing data, locating outliers, and producing illuminating low-dimensional views
# of the data.
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs = 8, n_estimators = 1000,oob_score =  True ,criterion = 'entropy')
# oob-  out of box sampling-no need for cross-validation or a separate test set to get an unbiased estimate of the test set error. It is estimated internally, during the run,
model.fit(x_train, y_train)

model.estimators_
model.classes_ # levels in class variable
model.n_classes_ #number of levels in class variable
model.n_features_ #number of features

#feature importance
fea_imp = pd.DataFrame(model.feature_importances_,
                       index = X.columns,
                       columns = ['importance']).sort_values('importance',ascending =False)
#US and Urban variables hold very little importance

model.score(x_train,y_train)
model.score(x_test,y_test) #0.9278
model.score(data.iloc[:,0:10],data.iloc[:,-1]) #0.9475
data['y_pred'] = model.predict(data.iloc[:,0:10])

from sklearn.metrics import confusion_matrix
cm = print(confusion_matrix(data.sales,data.y_pred))

np.mean(data.y_pred == data.sales) #0.9475
