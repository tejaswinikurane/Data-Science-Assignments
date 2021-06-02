# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:03:05 2020

@author: Admin
"""


import pandas as pd
import numpy as np

data = pd.read_csv('E:\\Tej\\Assignments\\Asgnmnt\\Random forest\\Fraud_check.csv')
data.describe()
data.info()
df = data['Taxable.Income']
data['status'] = np.where(df<=30000,'Risky','Good')
data.columns
data_cat=['Undergrad', 'Marital.Status','Urban', 'status']

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder
num = LabelEncoder()
for i in data_cat:
    data[i] = num.fit_transform(data[i])
 
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
X= df_unsampled.iloc[:,0:6]
Y= df_unsampled.iloc[:,-1]


#splitting into test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.3,random_state = 42)

#building Random forest classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs = 8,oob_score = True,n_estimators = 1500,criterion = 'entropy')
model.fit(x_train,y_train) #fiting the model on training dataset
model.classes_ #class labels
model.n_classes_ #number oof levels in class variable
model.n_features_ #nuber of features

model.score(x_train,y_train) #1.0
model.score(x_test,y_test)   #1.0

#storing results for original dataset
data['y_pred'] = model.predict(data.iloc[:,0:5])

#calculating feature importances
fea_imp = pd.DataFrame(model.feature_importances_,
                       index = X.columns,
                       columns = ['importance']).sort_values('importance',ascending = False)
#taxable income gives 93% information

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = print(confusion_matrix(data.y_pred,data.status))

#accuracy of model on original dataset
np.mean(data.y_pred == data.status)  #1.0

#accuracy of model on resampled dataset
model.score(X,Y) #1.0
