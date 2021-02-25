# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:10:33 2020

@author: Admin
"""


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing Dataset
data= pd.read_csv ('E:\\Tej\Assignments\\Asgnmnt\\Multi linear Regession\\50_Startups.csv')


#EDA
data.columns
data.info()

#converting object dtype to category for encoding
data['State']=data['State'].astype('category')

#replacing 0 with nan
from numpy import nan
data=data.replace(0,nan)
#replacing nan values with column means
data.fillna(data.mean(),inplace=True)

import seaborn as sns
sns.catplot(x='State',y='Profit',kind= 'box' ,data =  data) #newyork has outliers
sns.barplot( x='State',y='Profit',data = data)

sns.distplot(data['RndSpend'],kde=True)
sns.distplot(data['Marketing _Spend'],kde=True)
sns.distplot(data['Administration'],kde=True) #left skewed(negatively)

data.skew()
data.kurt()

sns.heatmap(data.corr(),annot=True)

#splitting into x and y
x=data.iloc[:,0:4].values
y=data.iloc[:,-1].values

#Encoding ctegorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
#[('kind of transformation,class of trans(),[column index])] passthrough=dont remove data which does not needs trans.
x=np.array(ct.fit_transform(x)) #transfroming features of matrix Xreturning new columns=no. of categories
 #our future ML model will expect the variable to be np array
print(x)

#splitting dataset into test train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=0)
#if random_state is mentioned same values in train and test datasets always

#building and training the model
from sklearn.linear_model import LinearRegression
model= LinearRegression()  #build the model
model.fit(x_train,y_train) #train the model
print(model.score(x_test,y_test)) #0.878

#predicting the results
y_pred=model.predict(x_test)
np.set_printoptions(precision=2) #upto 2 decimals
print(y_pred)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#prediction for the single value
print(model.predict([[1,0,0,160000,130000,300000]]))

print(model.coef_)    
print(model.intercept_)

#saving prediction in dataset
data['pred']=(model.predict(x))

#RMSE for Predicted data
y_resid=data.pred -data.Profit
y_rmse=np.sqrt(np.mean(y_resid*y_resid))
y_rmse #17950.72

#predicted vs actual
plt.scatter(data['Profit'],data['pred'])

data['pred'].corr(data['Profit']) #0.8938

from sklearn.metrics import r2_score
test_r2_score=r2_score(y_pred,y_test)    
print(test_r2_score)  #0.8612


####backward elimination####
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
import statsmodels.regression.linear_model as lm

#creating feature vector which only contain a set of independent variables
x_vtr=x[:,0:]
x_vtr=np.array(x_vtr,dtype=float)
model_be=lm.OLS(endog = y, exog = x_vtr).fit()
model_be.summary() #x5 is insignificant pvalue=0.995

x_vtr=x[:,[0,1,2,3,4,6]]
x_vtr=np.array(x_vtr,dtype=float)
model_be=lm.OLS(endog=y,exog=x_vtr).fit()
model_be.summary()  #x1 is insignificant, pvalue=0.092

x_vtr=x[:,[0,2,3,4,6]]
x_vtr=np.array(x_vtr,dtype=float)
model_be=lm.OLS(endog=y,exog=x_vtr).fit()
model_be.summary() #x2 is insignificant pvalue=0.374

x_vtr=x[:,[0,2,4,6]]
x_vtr=np.array(x_vtr,dtype=float)
model_be=lm.OLS(endog=y,exog=x_vtr).fit()
model_be.summary(()) #x1 is insignificant

x_vtr=x[:,[0,4,6]]
x_vtr=np.array(x_vtr,dtype=float)
model_be=lm.OLS(endog=y,exog=x_vtr).fit()
model_be.summary()

#therefore only Rndspend and Marketing-spend are significant, therefore we can now build model
#efficiently on these variables
dataset=data[['Profit','RndSpend','Marketing _Spend']]
dataset

x_be=dataset.iloc[:,1:].values
y_be=dataset.iloc[:,0].values

#splitting dataset into test and train
from sklearn.model_selection import train_test_split
xbe_train,xbe_test,ybe_train,ybe_test=train_test_split(x_be,y_be,test_size=0.2,random_state=0)


#training and building the model
from sklearn.linear_model import LinearRegression
model2=LinearRegression()
model2.fit(np.array(xbe_train),ybe_train)
model2.score(xbe_train,ybe_train)   #0.77
model2.score(xbe_test,ybe_test)     #0.90
model2.score(x_be,y_be)             #0.79   
