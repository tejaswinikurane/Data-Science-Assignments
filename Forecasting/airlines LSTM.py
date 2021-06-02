# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 08:58:35 2020

@author: Admin
"""
import pandas as pd

df = pd.read_excel('E:\\Tej\\Assignments\\Asgnmnt\\Forecasting\\Airlines_Data.xlsx')
df.head()
df.columns

df1 = df.reset_index()['Passengers']

df1.shape

import matplotlib.pyplot as plt
plt.plot(df1)


#LSTM values are sensitive to scale of data so we apply minmaxscaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

df1.shape

#train test split for time series data in ordered manner
train_size = int(len(df1)*0.65) #output should be integer value not float
test_size = len(df1)-train_size
train_data, test_data = df1[0:train_size,:], df1[0:test_size,:]

#converting data into dependent and independant using timesteps
#convert an array of values into dataset matrix
def create_dataset(dataset, time_step = 1):
    dataX, dataY = [],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]  ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)

time_step = 10
x_train,y_train = create_dataset(train_data,time_step)
x_test, y_test  = create_dataset(test_data,time_step)

print(x_train.shape)
print(y_train.shape)

#for LSTM model, x data needs to be in 3 dimensions[samples, timestep, festures]
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

#importing tf libarries for stacked LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

#create an stacked LSTM model
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape = (100,1))) #(timestep, feature)
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

model.summary()

model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 200, batch_size=64, verbose = 1)

#predicting the results
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

#rescaling the data to original form to get the desired results
train_predict = scaler.inverse_transform(train_predict)
test_predict  = scaler.inverse_transform((test_predict))

import math
from sklearn.metrics import mean_squared_error, accuracy_score
math.sqrt(mean_squared_error(y_train, train_predict)) #182.700
print(accuracy_score(y_train, train_predict))
math.sqrt(mean_squared_error(y_test, test_predict)) #151.947

### Plotting 
# shift train predictions for plotting
import numpy
look_back=10
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

#since, time_step =100, to predict the prices for next 30 days, we have to take values from last 100 days
x_input=test_data[86:].reshape(1,-1)
x_input.shape

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)