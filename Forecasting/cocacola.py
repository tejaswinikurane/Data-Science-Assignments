# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:17:08 2020

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('E:\\Tej\\Assignments\\Asgnmnt\\Forecasting\\CocaCola_Sales_Rawdata.xlsx')
df.head()
df.tail()

import seaborn as sns
sns.boxplot(df['Sales'])

df.set_index(['Quarter'], inplace = True)
df.dtypes
df.head()
df.describe()

df.plot();plt.xlabel('Quarter');plt.ylabel('sales')
#upward trend with addiive seasonality and non-stationary

#rolling statistics
rol_mean = df.rolling(window = 4).mean()
rol_sd = df.rolling(window = 4).std()

#plotting the data
plt.plot(df, label = 'original data')
plt.plot(rol_mean, color='red', label= 'rolling mean')
plt.plot(rol_sd, color = 'black',label = 'rolling std')
plt.legend(loc = 'best')
plt.show()

#augmented dickey-fuller test for stationarity
#H0- not sationary
#Ha -  stationary
from statsmodels.tsa.stattools import adfuller
test = adfuller(df['Sales'])
dfoutput = pd.Series(test[0:4], index = ['Test statistic', 'P-value','Lags used', 'No. of observations'])
for key,value in test[4].items():
    dfoutput['critical value (%s)'%key] = value
if dfoutput[1] <=0.05:
    print('strong evidence against null hypothsis(H0), reject null hypothesis, Data is not stationary')
else:
    print('weak evidence against null hypothsis(H0),Data is not stationary')
print(dfoutput)
# pvalue is too high<0.05, fail to reject H0, data is not stationary

#defining adfuller test function
def ts_adfuller(ts):
    from statsmodels.tsa.stattools import adfuller
    test1 = adfuller(ts['Sales'])
    dfoutput1 = pd.Series(test1[0:4], index = ['Test statistic', 'P-value','Lags used', 'No. of observations'])
    for key,value in test1[4].items():
        dfoutput1['critical value (%s)'%key] = value
    if dfoutput1[1] <=0.05:
        print('strong evidence against null hypothsis(H0), reject null hypothesis, Data is not stationary')
    else:
        print('weak evidence against null hypothsis(H0),Data is not stationary')
    print(dfoutput1)
    
dfshifted = df - df.shift(4)
dfshifted
dfshifted.dropna(inplace = True)
ts_adfuller(dfshifted)
#data is almost stationary

plt.plot(dfshifted)
from statsmodels.tsa.seasonal import seasonal_decompose
result  =  seasonal_decompose(df, model = 'additive', period = 12)
trend = result.trend
seasonal = result.seasonal
residual  = result.resid

plt.subplot(411)
plt.plot(df, label = 'original')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend, label = 'trend')
plt.legend(loc = 'best')
plt.subplot(413)
plt.plot(seasonal, label = 'seasonality')
plt.legend(loc = 'best')
plt.subplot(414)
plt.plot(residual, label = 'residuals')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

#from graph, residuals are not stationary


#determining p and q functions
from statsmodels.tsa.stattools import acf,pacf
acf_df = acf(df,nlags = 20, fft = False)
acf_pacf = pacf(df,nlags = 20,method = 'ols')

plt.plot(acf_df)
plt.axhline(y= 0, linestyle = '--')
plt.axhline(y = -1.96/np.sqrt(len(dfshifted)),linestyle = '--')
plt.axhline(y = 1.96/np.sqrt(len(dfshifted)),linestyle = '--')
plt.title('Autocorrelation plot with confidence intervals')
plt.show()

plt.plot(acf_pacf)
plt.axhline(y= 0, linestyle = '--')
plt.axhline(y = -1.96/np.sqrt(len(dfshifted)),linestyle = '--')
plt.axhline(y = 1.96/np.sqrt(len(dfshifted)),linestyle = '--')
plt.title('Parial Autocorrelation plot with confidence intervals')
plt.show()


