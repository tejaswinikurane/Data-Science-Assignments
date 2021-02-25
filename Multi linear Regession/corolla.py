# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 20:10:51 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('tableau-colorblind10')

#importing dataset
corolla= pd.read_csv('E:\\Tej\Assignments\\Asgnmnt\\Multi linear Regession\\ToyotaCorolla.csv',encoding='latin1')
corolla

#EDA
corolla.columns
corolla.head()
df=corolla[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]] #selecting given columns for operation

df.head() #top 5 obserations
df.info() #laast 5 operations
df.dropna() #drop na values #df.fillna(0) to fill na values with zero
df.describe()

import seaborn as sns
sns.barplot(x='Doors',y='Price',data=df)
sns.barplot(x='Gears',y='Price',data=df)

df['Gears'].value_counts().plot(kind='pie')
df['Doors'].value_counts().plot(kind='pie')

sns.distplot(df['Price'],kde=True)
sns.distplot(df['KM'],kde=True)

plt.scatter(df.Price,df.KM)
df.Price.corr(df.KM) #price decreases with km

sns.catplot(x='Doors',y='Price',data=df,kind='box')
sns.heatmap(df.corr(),annot=True)

#filling 1 values in KM column with nan and replacing with mean of the column
from numpy import nan
df.loc[df['KM']]=df['KM'].replace(1,nan)
df.fillna(df.mean(),inplace=True)
df.isna().sum() #0

import seaborn as sns
sns.pairplot(df)


#splitting data into x and y
y=df.iloc[:,0]
x=df.iloc[:,1:]


#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)


#since this is produced as a series object we've to convert it to np to produce results
y=y.to_numpy()
y_test=y_test.to_numpy()


#building model
from sklearn.linear_model import LinearRegression
model=LinearRegression()   #build the model
model.fit(x_train,y_train) #train the model
print(model.score(x_test,y_test))  #0.6555--> poor model

#predicting the results
y_pred= model.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

plt.scatter(y_pred,y_test)

#saving results in the dataset
df['pred']=(model.predict(x))

y_resid= y_pred-y_test

#standardized residuals
y_rmse= np.sqrt(np.mean(y_pred*y_test))

from sklearn.metrics import r2_score
test_r2s=r2_score(y_test,y_pred)

print(y_rmse)   #10895.54
print(test_r2s) #0.655



