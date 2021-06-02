# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 01:19:41 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import statsmodels.formula.api as smf

#Importing data set
data=pd.read_csv("E:\\Tej\\Assignments\\Asgnmnt\\Forecasting\\PlasticSales.csv")

month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
p=data['Month'][0]
data['month']=0
for i in range(60):
    p=data['Month'][i]
    data['month'][i]=p[0:3]

#EDA
data['Sales'].isnull().sum()
data['Sales'].mean() 
data['Sales'].median()
data['Sales'].mode()
data['Sales'].var()
data['Sales'].std()
data['Sales'].skew() #slight right skewed
data['Sales'].kurt() #slight flat curve
data.describe()

#getting dummies
month_dummies = pd.DataFrame(pd.get_dummies(data['month']))
data = pd.concat([data,month_dummies],axis = 1)

#creating new column for timeseries
#creating a new variable 't'
data['t']=np.arange(1,61)
#Creating a new variable 't_squared'
data["t_squared"] = data["t"]*data["t"]
#Creating a new variable 'log_Rider'
data["log_Rider"] = np.log(data["Sales"])

#Dropping Months column
data=data.drop('Month',axis=1)

#Splitting data into train and test data
Train = data.head(48)
Test = data.tail(12)

## Additive seasonality ##
add_sea = smf.ols('Sales~month',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['month']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea
#235.60
## Additive Seasonality Quadratic ##
add_sea_Quad = smf.ols('Sales~t+t_squared+month',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['month','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad #218.19

## Multiplicative Seasonality ##
Mul_sea = smf.ols('log_Rider~month',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea#239.654

## Multiplicative Additive Seasonality ##
Mul_Add_sea = smf.ols('log_Rider~t+month',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea #160.6833

#'Additive Seasonality Quadratic' model is working best with least rmse value
