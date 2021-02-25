# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:57:13 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#EDA
data=pd.read_csv('E:\\Tej\\Assignments\\Asgnmnt\PCA\\wine.csv')
data.describe()
data.info()
data.columns
data.shape
data.Type.value_counts()


#Data Preprocessing
from sklearn.preprocessing import LabelEncoder
number= LabelEncoder()
data['Type']=number.fit_transform(data['Type'])

def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return(x)

data.iloc[:,1:]=norm_func(data.iloc[:,1:])

#PCA
from sklearn.decomposition import PCA
pca=PCA()
pca_values=pca.fit_transform(data)

#amount of variance explained by each component
var= pca.explained_variance_ratio_
var
pca.components_

plt.plot(var) 

#cumulative sum if variances in %
var1=np.cumsum(np.round(var,decimals=3)*100) # we can use upto 7 PC's to get 95.1% information about data
plt.plot(var1,'co-')

x=pca_values[:,0]
y=pca_values[:,1]
z=pca_values[:,2]
plt.scatter(x,y);plt.xlabel('PC1');plt.ylabel('PC2') #no correlation

df1=pd.DataFrame(pca_values[:,:3])

#clustering on original dataset
data2=data.iloc[:,1:] #omitting first column 

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(data2)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss,'co-')
# optimum number of cluster=3


####Clustering on  using first 3 principal component scores####

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(df1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss,'ro-');plt.title('Scree Plot')
#optimum number of cluster = 3


import scipy.cluster.hierarchy as sch
z= sch.linkage(df1,method='ward',metric='euclidean')
sch.dendrogram(z);plt.title('dendrogram')



####hence, we have obtained same number of clusters with the original data and the first three primcipal components
#(class column we have ignored at the begining who shows it has 3 clusters)df
