# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:04:59 2020

@author: Tejaswini
"""


import pandas as pd
import numpy as np
from scipy import stats
import statsmodels as sm


data=pd.read_csv('E:\Tej\Assignments\Asgnmnt\Hypothesis\Cutlets.csv')
data
#Data are continuous and comparison of two population samples is to be done
####checking the data for normality####
#H0=data are normal and Ha=Data are not normal

unit_a=stats.shapiro(data['Unit A'])
unit_a_pvalue=unit_a[1]
print('p value is '+str(unit_a_pvalue))

#since p-value is > 0.05--> fail to reject null hypothesis 
#Data are normal

unit_b=stats.shapiro(data['Unit B'])
unit_b_pvalue= unit_b[1]
print('p value is '+str(unit_b_pvalue))
#since p-value is > 0.05--> fail to reject null hypothesis 
#Data are normal

####checking for variance####
#H0= Data have equal variance & Ha=Data have unequal variance
stats.levene(data['Unit A'],data['Unit B'])
#pvalue is >0.05, Fail to reject H0--> Data are having equal variance.
help(stats.levene)
#2 sample t_test
#H0= diameter of cutlet in Unit A= diameter of cutlet in Unit B
#Ha= diameter of cutlet in Unit A is not equal to the diameter of cutlet in Unit B
result=stats.ttest_ind(data['Unit A'],data['Unit B'])
print('p value is '+str(result[1]))
# since p-value is >0.05,Fail to reject H0
#Thus there is no any significant difference in the diameter of the cutlet between two units. 





####BuyerRatio#####
sales=pd.read_csv('D:\ML Docs\Excelr\Assignments\Hypothesis testing\BuyerRatio.csv',encoding='ISO-8859-1')
sales
sales.columns

sales1=sales.drop('Observed Values',axis=1)
sales1

sales1.values
stats.chi2_contingency(sales1)
#x=male,female  y=sales
#since both the x and y are discrete in 2+categories, we will go for chi-squared test
# Let H0= all proportiona are equal and Ha=Not all proportions are same.
chi2=stats.chi2_contingency(sales1)
chi2
chi2_pvalue=chi2[1]
print('p-value is '+str(chi2_pvalue))
help(stats.chi2_contingency)
#since, P-value=0.66>0.05-->P high Null fly. therefore, all proportions are equal.




####Average Turn Around Time####
#H0 = there is difference in the average Turn Around Time (TAT) of reports.
#Ha = there is no difference in the average Turn Around Time (TAT) of reports. 
tat=pd.read_csv('F:\\Excelr docs\\Assignments\\Hypothesis testing\\LabTAT.csv',encoding='ISO-8859-1')
tat
# here, x=no. of samples, y= Turn Around Time
# more than two samples are involved, therefore we will go for normality test.
#H0=Data are normal, Ha=Data are not Normal
lab1= stats.shapiro(tat['Laboratory 1'])
lab1_pvalue=lab1[1]
print('p-value is '+str(lab1_pvalue))
# pvalue=0.55,data are normal

lab2=stats.shapiro(tat['Laboratory 2'])
lab2_pvalue=lab2[1]
print('p-value is '+str(lab2_pvalue))
#pvalue=0.86, Data are normal

lab3=stats.shapiro(tat['Laboratory 3'])
lab3_pvalue=lab3[1]
print('p-value is '+str(lab3_pvalue))
#p-value=0.42, data are normal

lab4=stats.shapiro(tat['Laboratory 4'])
lab4_pvalue=lab4[1]
print('p-value is '+str(lab4_pvalue))
#p-value=0.66, data are normal

#since the data are normal, we will proceed for the variance test
# H0= data have equal variance, Ha=Data do not have equal Variance
var_lab1=stats.levene(tat['Laboratory 1'],tat['Laboratory 2'])
print('p value is '+str(var_lab1[1])) # pvalue=0.06 data have equal variance
# 
var_lab2=stats.levene(tat['Laboratory 1'],tat['Laboratory 3'])
print('p-value is '+str(var_lab2[1])) #pvalue=0.0064 data do not have equal variance

var_lab3=stats.levene(tat['Laboratory 1'],tat['Laboratory 4'])
print('p value is '+str(var_lab3[1])) #pvalue 0.221 data have equal variance

var_lab4=stats.levene(tat['Laboratory 2'],tat['Laboratory 3'])
print('p value is '+str(var_lab4[1])) # pvalue=0.33 data have equal variance

#one way ANOVA test
import scipy.stats as stats
outcome = stats.f_oneway(tat['Laboratory 1'],tat['Laboratory 2'],tat['Laboratory 3'],tat['Laboratory 4'])
p_value = outcome[1]
print(p_value)
# pvalue is less than 0.05, therefore, P low null go, accepting alternaate hypothesis that
# there is no difference in the average Turn Around Time (TAT) of reports.

####Faltoons####
data= pd.read_csv('D:\ML Docs\Excelr\Assignments\Hypothesis testing\Faltoons.csv',encoding='ISO-8859-1')
data
data.describe()
from sklearn.preprocessing import LabelEncoder
number= LabelEncoder()
data['Weekdays']=number.fit_transform(data['Weekdays'])
data['Weekend']=number.fit_transform(data['Weekend'])
data
#converted Male=0 and Female=0
data['Weekdays'].value_counts()
data['Weekend'].value_counts()

count= np.array([287,233]) # how many female and went for shopping om weedays and weekend
nos= np.array([400,400]) #how many total number of females went for shopping

#Ho= Male vs female walking to the store differ based on the days of week
#Ha= Male vs female walking to the store do not differ based on the days of week
from statsmodels.stats.proportion import proportions_ztest
stats,pval=proportions_ztest(count,nos)
print('{0:0.6f}'.format(pval))
#P-value-->0.00006<0.05. P low null go
#therefore accepting Alternative hypothesis
#so, Male vs female walking to the store do not differ based on the days of week






