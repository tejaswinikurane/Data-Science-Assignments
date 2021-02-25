# -*- coding: utf-8 -*-
"""
Created on Wed Sep 01 13:09:26 2021

@author: Admin



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

airlines=pd.read_excel('E:\Tej\Assignments\Asgnmnt\Clustering\EastWestAirlines.xlsx',sheet_name='data')
airlines
airlines.info()

def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return(x)

y=norm_func(airlines.iloc[:,[1,6,7,8,9]])

z=airlines.iloc[:,[2,3,4,5,10,11]]

data=pd.concat([y,z],axis=1)

data.replace(np.nan,0,inplace= True)
data.info()
data.describe()

####K-means Clustering####
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters= i,init='k-means++',random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss,'ro-');
plt.xlabel('No. of Clusters');
plt.ylabel('wcss');
plt.title('Scree plot')
#from 4 the curve started flattening, therefore we choose 4 as optimum no. of clusters

kmeans_f=KMeans(n_clusters=4,init='k-means++',random_state=42)
y_kmeans=kmeans_f.fit_predict(data)    
y_kmeans
data['cluster']=kmeans_f.labels_
airlines['cluster']=kmeans_f.labels_
airlines.cluster.value_counts()
#cluster 0 is of small group of premium customers-53
#cluster 2 & 3 consists of middle spending customers
#cluster 1 is having least spending customers

cluster1= data.loc[y_kmeans == 0]
cluster2= data.loc[y_kmeans == 1]
cluster3= data.loc[y_kmeans == 2]
cluster4= data.loc[y_kmeans == 3]
                     
sns.barplot(x='cluster',y='Balance',data=airlines)           #order=1,3,2,0 lowest to highest
sns.barplot(x='cluster',y='Qual_miles',data=airlines)        #order=1,3,2,0    
sns.barplot(x='cluster',y='cc1_miles',data=airlines)         #order=1,0,3,2
sns.barplot(x='cluster',y='cc2_miles',data=airlines)         
sns.barplot(x='cluster',y='cc3_miles',data=airlines)         
sns.barplot(x='cluster',y='Bonus_miles',data=airlines)       #order=1,3,0,2
sns.barplot(x='cluster',y='Bonus_trans',data=airlines)       #order=1,3,2,0
sns.barplot(x='cluster',y='Flight_miles_12mo',data=airlines) #order=1,3,2,0
sns.barplot(x='cluster',y='Flight_trans_12',data=airlines)   #order=1,3,2,0
sns.barplot(x='cluster',y='Days_since_enroll',data=airlines) #order=1,0,3,2
sns.barplot(x='cluster',y='Award?',data=airlines)            #order=1,3,2,0
#cluster 0 are high spending cluster as it has highest number of miles eligible for free travel and rewards also alongwith other parameters
#cluster 2 & 3 are middle spending clusters with average balance and rewards and offers.
#cluster 1 is of least spending customers with least flight miles and miles eligible for free travel
#customers in cluster 2&3 are enrolled for long days than others and cluster 0 are in the middle of a & 2-3.
#customers in cluster 0 have received most awards due to their high spendings and that of cluster 1 have received the least.
#for customers with least flight transactions(cluster 1), discounted fair rates should be given to increase their spending and the to make them to stick to the airlines.

plt.scatter(cluster1['Balance'],cluster1['Bonus_miles'],c='red',label='cluster1')
plt.scatter(cluster2['Balance'],cluster2['Bonus_miles'],c='blue',label='cluster2')
plt.scatter(cluster3['Balance'],cluster3['Bonus_miles'],c='magenta',label='cluster3')
plt.scatter(cluster4['Balance'],cluster4['Bonus_miles'],c='black',label='cluster4')
plt.scatter(kmeans_f.cluster_centers_[:,0],kmeans_f.cluster_centers_[:,1],c='cyan',label='centroids')
plt.show()

data.iloc[:,:].groupby('cluster').mean()

sns.pairplot(data,hue='cluster')
                                          
                                          
####Hierarchcal clustering ####
import scipy.cluster.hierarchy as sch
z=sch.linkage(data, method = 'ward',metric='euclidean')
plt.figure(figsize=(15,10));plt.title('Hierarchical Clustering Dendrogram')
sch.dendrogram(z,
               leaf_rotation=0.,);plt.show()

#alternative method
from sklearn.cluster import AgglomerativeClustering
h_cluster= AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward').fit(data)                                         

data['cluster']=h_cluster.labels_
data.iloc[:,0:].groupby('cluster').mean()

data.to_csv('EastWestAirlines.csv',encoding='utf-8')
