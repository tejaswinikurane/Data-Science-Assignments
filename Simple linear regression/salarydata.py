# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:30:05 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import scipy.stats as stats

#importing dataset
df= pd.read_csv('E:\Tej\Assignments\Asgnmnt\Simple linear regression\Salary_Data.csv',encoding='ISO-8859-1')
df

df.columns
df.head()
df.tail()

df.info() #no null values
df.describe()

plt.hist(df.YearsExperience) #Data not normal
plt.plot(df.YearsExperience,'ro')

plt.hist(df.Salary) #Data not normal
plt.plot(df.Salary,'ro')

prob_YE= stats.shapiro(df['YearsExperience'])
prob_YE
#pvalue=0.1033>0.05, data is normal

prob_sal=stats.shapiro(df['Salary'])
prob_sal
#pvalue=0.015<0.05,data is not normal

df['Salary'].corr(df['YearsExperience'])
#strong correlation 0.978
np.corrcoef(df['Salary'],df['YearsExperience'])

#building model
model1= smf.ols('Salary~YearsExperience',data=df).fit()
model1.summary() #R-squared=0.957 ,strong model

pred=model1.predict(pd.DataFrame(df['YearsExperience']))
pred

df['Predicted_Salary']= pred
df

#residuals
resid= pred-df['Salary']
resid

#finding standardized residuals
student_resid=model1.resid_pearson
student_resid

plt.plot(student_resid,'ro');plt.axhline(y=0);plt.xlabel('Observed Values');plt.ylabel('standardized error')

#best fit line of the model
plt.scatter(df['YearsExperience'],df['Salary']);plt.plot(df['YearsExperience'],pred,color='blue');plt.xlabel('YearsExperience');plt.ylabel('Salary')

#predicted vs Actual values
plt.plot(pred,df['Salary'],'bo'); xlabel('Actual Salary');ylabel('Predicted Salary')
