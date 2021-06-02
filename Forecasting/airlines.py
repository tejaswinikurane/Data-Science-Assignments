# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:43:48 2020

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('E:\\Tej\\Assignments\\Asgnmnt\\Forecasting\\Airlines_Data.xlsx')
data.head()
data.columns

month_index = data.set_index(['Month'])

plt.plot(month_index);plt.xlabel('Month');plt.ylabel('No. of Passengers')
#data having upward trend with multiplicative seasonality, non stationary
month_index.head()
month_index.tail()

#rolling statistics
rolling_mean = month_index.rolling(window = 12).mean() #window = 12 for 12 months, for days=365
rolling_std = month_index.rolling(window = 12).std()
print(rolling_mean)
print(rolling_std)

original = plt.plot(month_index, color = 'blue', label = 'original')
mean = plt.plot(rolling_mean, color = 'black',label = 'rolling mean')
std = plt.plot(rolling_std, color = 'red', label = 'rolling std')
plt.legend(loc = 'best')
plt.title('rolling mean and rolling std')
plt.show()

#perform dickey-fuller test to check stationarity of data
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(month_index['Passengers'],autolag = 'AIC')
dfoutput = pd.Series(dftest[0:4], index = ['Test statistic','P-value','Lags-used','Number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)'%key] = value
    
print(dfoutput)
#since p-value >0.05, accept null hypothesis that unit root is present in AR model and data is not stationary. 

#estimating trend
month_index_log = np.log(month_index)   
plt.plot(month_index_log)
#upward trend remains same, but value of y has been changed

#rolling statistics for transformed data
moving_average = month_index_log.rolling(window = 12).mean()
moving_std = month_index_log.rolling(window = 12).std()
print(moving_average)
print(moving_std)

orig = plt.plot(month_index_log,color = 'blue',label = 'log transformed')
mean_log = plt.plot(moving_average,color = 'black',label = 'rolling mean_log')
plt.legend(loc = 'best')
plt.title('moving average')
plt.show()
#upward trend still persists

#getting difference between log values and moving average
logMinusMA = month_index_log - moving_average
logMinusMA
logMinusMA.dropna(inplace = True)
logMinusMA.head()

#stationarity check function
from statsmodels.tsa.stattools import adfuller
def test_stat(ts):
    
    #determining rolling statistics
    mov_avg = ts.rolling(window = 12).mean()
    mov_std = ts.rolling(window = 12).std()
    
    #plotting rolling statistics 
    org = plt.plot(ts,color = 'blue',label = 'original')
    MA = plt.plot(mov_avg, color = 'red',label = 'Moving average')
    std = plt.plot(mov_std, color = 'black',label ='Moving std')
    plt.legend(loc = 'best')
    plt.title('rolling statistics')
    plt.show()
    
    #performing Dickey-fuller test for sationarity check
    df_test = adfuller(ts['Passengers'],autolag = 'AIC')
    df_output = pd.Series(df_test[0:4],index = ['Test statistic', 'p-value','#Lags used','No. of Observations used'])
    for key, value in df_test[4].items():    #.items is very important
        df_output['critical value (%s)'%key] = value
    print(df_output)
    

test_stat(logMinusMA)
#p-value<0.05, and test statistic is approx.= critical value.therefore, time series is now stationary

plt.plot(logMinusMA)

#differentiating
logMinusMAshifted = month_index_log - month_index_log.shift()
plt.plot(logMinusMAshifted)

logMinusMAshifted.dropna(inplace = True)
test_stat(logMinusMAshifted)
#since p-value is nearly = 0.05, data is almost stationary. 

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(month_index_log, model = 'additive', period = 12)
trend = result.trend 
seasonal = result.seasonal
residual = result.resid

plt.subplot(411)
plt.plot(month_index_log, label = 'original')
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

#checking residuals for stationarity
dec_log = residual
dec_log.dropna(inplace = True)

 
mov_avg1 = dec_log.rolling(window = 12).mean()
mov_std1 = dec_log.rolling(window = 12).std()
    
#plotting rolling statistics 
xorg1 = plt.plot(dec_log,color = 'blue',label = 'original')
MA1 = plt.plot(mov_avg1, color = 'red',label = 'Moving average')
std1 = plt.plot(mov_std1, color = 'black',label ='Moving std')
plt.legend(loc = 'best')
plt.title('noise component')
plt.show()
    
dec_log = dec_log.to_frame()
#performing Dickey-fuller test for sationarity check
df_test1 = adfuller(dec_log['resid'],autolag = 'AIC')
df_output1 = pd.Series(df_test1[0:4],index = ['Test statistic', 'p-value','#Lags used','No. of Observations used'])
for key, value in df_test1[4].items():    #.items is very important
    df_output1['critical value (%s)'%key] = value
print(df_output1)
# Noise component not stationary

#d = 1
#plotting acf plot for q and for value of p plotting pacf plot
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(logMinusMAshifted,nlags = 20,fft = False)
lag_pacf = pacf(logMinusMAshifted,nlags = 20, method = 'ols')

#plot acf
plt.plot(lag_acf)
plt.axhline(y = 0,linestyle='--')
plt.axhline(y = -1.96/np.sqrt(len(logMinusMAshifted)),linestyle = '--')
plt.axhline(y = 1.96/np.sqrt(len(logMinusMAshifted)),linestyle = '--')
plt.title('Autocorrelation plot')
#graph first approaches to zero at approx. 2 so, q=2

#plot pacf plot
plt.plot(lag_pacf)
plt.axhline(y = 0,linestyle='--')
plt.axhline(y = -1.96/np.sqrt(len(logMinusMAshifted)),linestyle = '--')
plt.axhline(y = 1.96/np.sqrt(len(logMinusMAshifted)),linestyle = '--')
plt.title('Partial Autocorrelation plot')
#graph first approaches to zero at approx. 2 so, p=2

#AR model(taking q = 0)
from statsmodels.tsa.arima_model import ARIMA
model_ar = ARIMA(month_index_log, order = (2,1,0))
result_ar = model_ar.fit(disp = -1 )

plt.plot(logMinusMAshifted)
plt.plot(result_ar.fittedvalues, color = 'red')
plt.title('RSS: %.4f'% sum((result_ar.fittedvalues - logMinusMAshifted['Passengers'])**2))
#RSS-0.9508
 
#MA model (taking p = 0)
model_ma = ARIMA(month_index_log, order = (0,1,2))
result_ma = model_ma.fit(disp = -1 )

plt.plot(logMinusMAshifted)
plt.plot(result_ma.fittedvalues, color = 'red')
plt.title('RSS: %.4f'% sum((result_ma.fittedvalues - logMinusMAshifted['Passengers'])**2))
#RSS-0.8278
 
#combined ARIMA model for forecasting
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(month_index_log, order = (2,1,2))
result = model.fit(disp = -1 )

plt.plot(logMinusMAshifted)
plt.plot(result.fittedvalues, color = 'red')
plt.title('RSS: %.4f'% sum((result.fittedvalues - logMinusMAshifted['Passengers'])**2))
#RSS-0.6931
 

#predictions
predictions = pd.Series(result.fittedvalues,copy = True)
print(predictions.head())

predictions_cumsum = predictions.cumsum()
print(predictions_cumsum.head())

pred_log = pd.Series(month_index_log.iloc[:,0], index = month_index_log.index)
pred_log = pred_log.add(predictions_cumsum, fill_value = 0)
print(pred_log.head())

pred_arima = np.exp(pred_log)

plt.plot(month_index, color = 'blue')
plt.plot(pred_arima, color = 'red')
plt.title('RMSE: %.4f'% np.sqrt(sum((pred_arima-month_index.iloc[:,0])**2)/len(month_index)))

result.plot_predict(1,200)
x = result.forecast(steps = 120)

