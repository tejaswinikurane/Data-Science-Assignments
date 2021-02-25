# -*- coding: utf-8 -*--
"""
Created on Wed Aug 19 23:01:37 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data= pd.read_csv("E:\\Tej\Assignments\\Asgnmnt\\Logistic regression\\bank-full.csv",delimiter=';')
data.head()

#changing unknown values with mode of the data(imputation)
from sklearn.preprocessing import LabelEncoder
number= LabelEncoder()
data.job=number.fit_transform(data.job)
data.marital=number.fit_transform(data.marital)
data.education=number.fit_transform(data.education)
data.default=number.fit_transform(data.default)
data.housing=number.fit_transform(data.housing)
data.loan=number.fit_transform(data.loan)
data.month=number.fit_transform(data.month)
data.poutcome=number.fit_transform(data.poutcome) 
data.y=number.fit_transform(data.y) 


#dropping irrelevant variables
sns.countplot(x='marital',hue='y',data=data)
sns.countplot(x='education',hue='y',data=data)
sns.countplot(x='default',hue='y',data=data) # drop- skewed towards zero
sns.countplot(x='housing',hue='y',data=data)  
sns.countplot(x='loan',hue='y',data=data)  #drop=skewed towards zero
sns.countplot(x='contact',hue='y',data=data) #insignificant for the prediction model

data.y.value_counts() #0-39922,1-5289
#variables for building the model
cat_var= data[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome']]
data.columns
data.isnull().sum()
cat_var.mode()
cat_var.skew()
cat_var.kurt()
cat_var.shape
cat_var.nunique()
cat_var.dropna().shape
cat_var['education'].value_counts()
cat_var['poutcome'].value_counts() #0-39922,1-5289 this imbalance in data will lead to bias.


for col in cat_var:
    plt.figure(figsize=(10,4))
    sns.barplot(cat_var[col].value_counts(),cat_var[col].value_counts().index)
    plt.title(col)
    plt.tight_layout()


#to remove the entire row containing na values #data= data[data.job!='unknown']

pd.crosstab(data.job,data.marital).plot(kind='bar')
pd.crosstab(data.job,data.default).plot(kind='bar')
pd.crosstab(data.housing,data.loan).plot(kind='bar')
pd.crosstab(data.education,data.default).plot(kind='bar')
pd.crosstab(data.housing,data.marital).plot(kind='bar')
#data are imbalanced, so we will resample the data to get the balanced data for classification

##oversampling
majority_class = cat_var[cat_var.poutcome == 0]
minority_class = cat_var[cat_var.poutcome == 1]

from sklearn.utils import resample
minority_class_unsampled = resample(minority_class,
                                    replace = True, #sample with replacement
                                    n_samples = 39922, #to match majority class
                                    random_state = 123) #reproduce results

data_resampled = pd.concat([majority_class,minority_class_unsampled])

#separating dependent andindependent variables
X= data_resampled.iloc[:,0:14]
Y= data_resampled.iloc[:,-1]

#splitting into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
y_train.value_counts().nunique()
y_test.value_counts()

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
classifier.score(x_test,y_test)
classifier.score(x_train,y_train) #0.762
classifier.score(cat_var.iloc[:,0:14],cat_var.iloc[:,-1]) #0.789

#getting coefficients and intercepts
classifier.coef_
classifier.intercept_

#getting results on test data
predictions=classifier.predict(x_test)

#gerring predictions on original data
y_prob=pd.DataFrame(classifier.predict(cat_var.iloc[:,0:14]))
new_df= pd.concat([cat_var,y_prob],axis=1)

#accurcy on original data
np.mean(cat_var.poutcome == new_df[0]) #0.789

pd.crosstab(new_df.poutcome,new_df[0]).plot(kind = 'bar')

from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,predictions)
confusion_matrix

#accuracy on test data
np.mean(y_test == predictions) #0.7600
