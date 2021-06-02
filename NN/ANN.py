# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 08:44:06 2020

@author: Admin
"""


import pandas as pd
import numpy as np


df = pd.read_csv("E:\\Tej\\Assignments\\Asgnmnt\\NN\\forestfires.csv")

#As dummy variables are already created, we will remove the month and and columns
df.drop(["month","day"],axis=1,inplace = True)

df["size_category"].value_counts()
df.isnull().sum()
df.describe()

##small = 0, large = 1
df.loc[df["size_category"]=='small','size_category']=0
df.loc[df["size_category"]=='large','size_category']=1
df["size_category"].value_counts()

#Defining Normalization function
def norm_func(i):
     x = (i-i.min())/(i.max()-i.min())
     return (x)

#separating predictors and target variables
predictors = df.iloc[:,0:28]
target = df.iloc[:,28]

#normalizing the predictors
predictors1 = norm_func(predictors)
#data = pd.concat([predictors1,target],axis=1)

#splitting the data into train and test datasetss
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(predictors1,target, test_size=0.3,stratify = target)


from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda
def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1],kernel_initializer="normal",activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer = "rmsprop",metrics = ["accuracy"])
    return model  

#y_train = pd.DataFrame(y_train)

#building the model
first_model = prep_model([28,50,40,20,1])
first_model.fit(np.array(x_train),np.array(y_train),epochs=500)
pred_train = first_model.predict(np.array(x_train))

#Converting the predicted values to series 
pred_train = pd.Series([i[0] for i in pred_train])

#predictions of class for train data
size = ["small","large"]
pred_train_class = pd.Series(["small"]*361)
pred_train_class[[i>0.5 for i in pred_train]]= "large"

train = pd.concat([x_train,y_train],axis=1)
train["size_category"].value_counts()

#metrics of success For training data
from sklearn.metrics import confusion_matrix
train["original_class"] = "small"
train.loc[train["size_category"]==1,"original_class"] = "large"
train.original_class.value_counts()
confusion_matrix(pred_train_class,train["original_class"])
np.mean(pred_train_class==pd.Series(train["original_class"]).reset_index(drop=True)) #100%
pd.crosstab(pred_train_class,pd.Series(train["original_class"]).reset_index(drop=True))

#metrics of success For test data
pred_test = first_model.predict(np.array(x_test))
pred_test = pd.Series([i[0] for i in pred_test])
pred_test_class = pd.Series(["small"]*156)
pred_test_class[[i>0.5 for i in pred_test]] = "large"
test =pd.concat([x_test,y_test],axis=1)
test["original_class"]="small"
test.loc[test["size_category"]==1,"original_class"] = "large"

test["original_class"].value_counts()
np.mean(pred_test_class==pd.Series(test["original_class"]).reset_index(drop=True)) # 89.10%
confusion_matrix(pred_test_class,test["original_class"])
pd.crosstab(pred_test_class,pd.Series(test["original_class"]).reset_index(drop=True))
