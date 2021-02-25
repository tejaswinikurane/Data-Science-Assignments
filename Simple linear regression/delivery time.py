# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:25:41 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

data= pd.read_csv('E:\Tej\Assignments\Asgnmnt\Simple linear regression\delivery_time.csv',encoding='ISO-8859-1')
data

data.describe()
data.info()

data.head()
data.tail()

data.columns

sns.pairplot(data)

plt.hist(data['Delivery_Time']) #data are not normal
help(plt.boxplot)
plt.boxplot(data['Delivery_Time'],vert=True, patch_artist=True)

plt.hist(data['Sorting_Time']) # data are not normal
plt.boxplot(data['Sorting_Time'],0,'rs',vert=True,patch_artist=True)

plt.plot(data['Delivery_Time'],data['Sorting_Time'],'co');plt.xlabel('Delivery_Time');plt.ylabel('Sorting_Time');plt.title('Scatterplot')
help(plt.plot)

data['Delivery_Time'].corr(data['Sorting_Time']) #0.8259 moderate correlation
np.corrcoef(data['Delivery_Time'],data['Sorting_Time'])

#building model
model1= smf.ols('data.iloc[:,0]~data.iloc[:,1]',data=data).fit()
model1.summary() #R-squared=0.682

#transforming variables for accuracy
#model2 = smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2 = smf.ols('Delivery_Time ~ np.log(Sorting_Time)',data=data).fit()
model2.summary() #R squared improved to 0.711

#again transforming the model
model3= smf.ols('np.log(Delivery_Time)~np.log(Sorting_Time)',data=data).fit()
model3.summary()  # R squared 0.772 this is not a strong model since R-squared<0.8
# so, we will consider this model aas the final model with highest R-squared value

pred_3= model3.predict(pd.DataFrame(data['Sorting_Time']))
pred_3.corr(data['Sorting_Time'])
pred_3
pred3=np.exp(pred_3)
pred3
pred3.corr(data['Sorting_Time'])

data['predicted']=pred3
data
plt.scatter(data['Sorting_Time'],data['Delivery_Time']);plt.plot(data['Sorting_Time'],pred3,color='blue');plt.xlabel('Sorting_Time');plt.ylabel('Delivery_Time')

resid=pred3-data['Delivery_Time']
resid
#residuals of entire dataset
student_resid= model3.resid_pearson
student_resid
plt.plot(student_resid,'o');plt.axhline(y=0,color='green');plt.xlabel('Observed numbers');plt.ylabel('standardized residuals')
plt.hist(student_resid)

#predicted Vs. Actual values
plt.scatter(pred3,data.Delivery_Time,color='red');plt.xlabel('predicted');plt.ylabel('actual')

model3.conf_int(0.05)
