# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 11:43:38 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#imporing dataset
df= pd.read_csv('E:\Tej\Assignments\Asgnmnt\Multi linear Regession\Computer_Data.csv',encoding='ISO-8859-1')

#EDA
df.head()
df.columns
df.info()

df1=df[['price', 'speed', 'hd']]
sns.barplot(x='ram',y='price',data=df)
sns.barplot(x='cd',y='price',data=df)
sns.barplot(x='ads',y='price',data=df)
plt.scatter('hd','speed',data=df)

df.skew()
df.kurt()

sns.heatmap(df1.corr(),annot=True)
sns.distplot(df['price'],kde=True) #positively skewed
sns.distplot(df['hd'],kde=True) #not normal 

sns.catplot(x='screen',y='price',data=df,kind='box')
sns.catplot(x='screen',y='price',data=df,kind='box')

plt.plot(df.price,df.ram,'ro')
df.ram.value_counts().plot(kind='pie')

df.price.groupby(df.screen).plot(kind='hist')

#ENcoding categorical data
from sklearn.preprocessing import LabelEncoder
number= LabelEncoder()
df['cd']= number.fit_transform(df['cd'])
df['multi']= number.fit_transform(df['multi'])
df['premium']= number.fit_transform(df['premium'])
df.dtypes

#separating x and y variables
x= df.iloc[:,1:].values
y=df.iloc[:,0].values

sns.pairplot(df)


#splitting data into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#creating linear model
from sklearn.linear_model import LinearRegression
model=LinearRegression() #model Building
model.fit(x_train,y_train) #model trainig

#predicting the results
y_pred= model.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#scatterplot of predicted vs actual values 
plt.scatter(y_pred,y_test)

#saving the results in dstaframe
df['pred']=model.predict(x)

#correlation between actual and predicted values
df['price'].corr(df['pred']) #0.8806 

#RMSE value for test data
test_rmse =np.sqrt(np.mean(y_test*y_pred))#2260.56

#R^2 value
from sklearn.metrics import r2_score
test_r2_Score= r2_score(y_test,y_pred) #0.8612

print(test_rmse)    #2260.56
print(test_r2_Score)#0.7743


#Backward elimintion#
x=np.append(arr=np.ones((6259,1)).astype(int),values=x,axis=1)
import statsmodels.regression.linear_model as lm
x_opt=x[:,0:]
model_be=lm.OLS(endog=y,exog=x_opt).fit()
model_be.summary() #since, every variable is having pvalue<0.05,we'll consider the model as final model.
