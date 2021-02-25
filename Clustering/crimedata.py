# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 08:56:57 2020

@author: Admin
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cd=pd.read_csv('E:\Tej\Assignments\Asgnmnt\Clustering\crime_data.csv')
cd

cd.info()
cd.describe()
cd.columns


def norm_func(i):
    x= (i- i.min())/(i.max() - i.min())
    return(x)

X=norm_func(cd.iloc[:,1:])

####KMeans Clustering####

from sklearn.cluster import KMeans
wcss = [] #store wcss values for each cluster in list
for i in range(1,11): #no. of clusters
    kmeans= KMeans(n_clusters= i, init='k-means++',random_state=42) #init to avoid random initialization trap
    kmeans.fit(X)  #trained and run kmeans algorithm
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss,'ro-');plt.title('ELbow Method');plt.xlabel('Number of Clusters');plt.ylabel('wcss');plt.show()
# we must choose 4 as the optimal number of clusters as the curve starts flattening from 4

kmeans_f=KMeans(n_clusters=4,init='k-means++',random_state=42)
y_kmeans=kmeans_f.fit_predict(X) #predicting the cluster for each row
y_kmeans
kmeans_f.labels_

cd['clusters']= kmeans_f.labels_  #storing in the dataset
cd.iloc[:,:].groupby(cd.clusters).mean()

cluster1=X.loc[y_kmeans == 0]
cluster2=X.loc[y_kmeans == 1]
cluster3=X.loc[y_kmeans == 2]
cluster4=X.loc[y_kmeans == 3]

plt.scatter(cluster1['Murder'],cluster1['Assault'], c='red',label='Cluster 1')
plt.scatter(cluster2['Murder'],cluster2['Assault'], c='blue',label='Cluster 2')
plt.scatter(cluster3['Murder'],cluster3['Assault'], c='cyan',label='Cluster 3')
plt.scatter(cluster4['Murder'],cluster4['Assault'], c='black',label='Cluster 4')
plt.scatter(kmeans_f.cluster_centers_[:,0],kmeans_f.cluster_centers_[:,1],c='magenta',label='Centroids')
plt.title('Clusters of crime')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.show()

sns.pairplot(cd,hue='clusters')

sns.barplot(x='clusters',y='Murder',data=cd)   #order=2,0,1,3
sns.barplot(x='clusters',y='Assault',data=cd)  #order=2,0,1,3
sns.barplot(x='clusters',y='UrbanPop',data=cd) #order=2,3,0,1
sns.barplot(x='clusters',y='Rape',data=cd)     #order=2,0,3,1

#cluster 2 is least vulnerable to the crimes and has lowest rates of crime and also has least urban population
#Even if cluster 0 is having 2nd highest urban population, it has least rate of crimes after cluster2
#cluster 1 having highest urban population, has highest rate of rapes and is second in terms of murder and assaults
#cluster 3 is having lower Urban population, but it is having highest number of Murder and Assault crimes.


####Hierarchicl clustering ####
from sklearn.cluster import AgglomerativeClustering
hclust= AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward').fit(X)
y_hc= hclust.fit_predict(X)
cd['clusters']=hclust.labels_

cd.iloc[:,1:].groupby('clusters').mean()

import scipy.cluster.hierarchy as sch
z= sch.linkage(X,method='ward',metric='euclidean')
sch.dendrogram(z);plt.xlabel('state index');plt.ylabel('Euclidean distance');plt.title('Dendrogram')

cluster1=X.loc[y_hc == 0]
cluster2=X.loc[y_hc == 1]
cluster3=X.loc[y_hc == 2]
cluster4=X.loc[y_hc == 3]

plt.scatter(cluster1['Murder'],cluster1['Assault'], c='red',label='Cluster 1')
plt.scatter(cluster2['Murder'],cluster2['Assault'], c='magenta',label='Cluster 2')
plt.scatter(cluster3['Murder'],cluster3['Assault'], c='cyan',label='Cluster 3')
plt.scatter(cluster4['Murder'],cluster4['Assault'], c='blue',label='Cluster 4')
plt.title('Clusters of crime')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.show()

