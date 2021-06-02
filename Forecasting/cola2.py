# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:25:42 2020

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('E:\\Tej\\Assignments\\Asgnmnt\\Forecasting\\CocaCola_Sales_Rawdata.xlsx')
df.head()
df.tail()

import seaborn as sns
sns.boxplot(df['Sales'])

df.set_index(['Quarter'], inplace = True)
df.dtypes
df.head()
df.describe()

# Lets us use auto_arima from p
from pmdarima import auto_arima
auto_arima_model = auto_arima(df['Sales'],start_p=0,
                              start_q=0,max_p=5,max_q=5,
                              m=12,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=False)
#ARIMA(4,1,0)(1,1,0)[12] : AIC=400.949
auto_arima_model.summary()

# Using Sarimax from statsmodels 
# As we do not have automatic function in indetifying the 
# best p,d,q combination 
# iterate over multiple combinations and return the best the combination
# For sarimax we require p,d,q and P,D,Q 
from products.models import Product

combinations_l = list(product(range(1,7),range(2),range(1,7)))
combinations_u = list(product(range(1,7),range(2),range(1,7)))
m =12 

results_sarima = []
best_aic = float("inf")

for i in combinations_l:
    for j in combinations_u:
        try:
            model_sarima = sm.tsa.statespace.SARIMAX(df["Sales"],
                                                     order = i,seasonal_order = j+(m,)).fit(disp=-1)
        except:
            continue
        aic = model_sarima.aic
        if aic < best_aic:
            best_model = model_sarima
            best_aic = aic
            best_l = i
            best_u = j
        results_sarima.append([i,j,model_sarima.aic])

df.plot();plt.xlabel('Quarter');plt.ylabel('sales')
#upward trend with addiive seasonality and non-stationary

#rolling statistics
rol_mean = df.rolling(window = 4).mean()
rol_sd = df.rolling(window = 4).std()

#plotting the data
plt.plot(df,color = 'blue', label = 'original data')
plt.plot(rol_mean, color='red', label= 'rolling mean')
plt.plot(rol_sd, color = 'black',label = 'rolling std')
plt.legend(loc = 'best')
plt.show()

df_shifted = pd.concat([df, df.shift(4)],axis  = 1)

df_shifted.columns  = ['actual','shifted']
df_shifted  = df_shifted.dropna()

from sklearn.metrics import mean_squared_error
df_rmse = np.sqrt(mean_squared_error(df_shifted.actual,df_shifted.shifted ))
print(df_rmse) #363.783

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(df)
# q = 4
plot_pacf(df)
# p = 1

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df,order = (3,1,1))
model.fit()

#best possible values for p d qubing trial and erroe method
pv = range(0,4)
dv = range(0,3)
qv = range(0,3)

for p in pv:
    for d in dv:
        for q in qv:
            order = (p,d,q)
            pred = list()
            for i in range(len(df)):
                try:
                    model = ARIMA(df, order)
                    result  = model.fit(disp = 0)
                    pred_y = result.forecast()[0]
                    pred.append(pred_y)
                    error = mean_squared_error(df,pred)
                    print('arima %s MSE = %.2f'% (order, error))
                except:
                    continue
aa = print('arima %s MSE = %.2f'% (order, error))



