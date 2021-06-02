# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 19:48:52 2021

@author: Admin
"""

import pandas as pd
import numpy as np

book_df= pd.read_csv("E:\\Tej\\Assignments\\Asgnmnt\\Recommandation\\book.csv",encoding=('ISO-8859-1'))
book_df[0:5]
book_df=book_df.drop('Unnamed: 0',axis=True)
book_df=book_df.rename({'User.ID':'UserID','Book.Title':'BookTitle','Book.Rating':'BookRating'},axis=1)
book_df
len(book_df.UserID.unique())
len(book_df.BookTitle.unique())
import seaborn as sns
sns.pairplot(book_df)
user_book_df=book_df.pivot_table(index='UserID', columns='BookTitle', values='BookRating',aggfunc='mean').reset_index(drop=True)
user_book_df
user_book_df = book_df.groupby(['UserID', 'BookTitle'])['BookRating'].mean().unstack()
user_book_df
user_book_df.index = book_df.UserID.unique()
user_book_df
user_book_df.fillna(0, inplace=True)
user_book_df
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation
user_sim= 1 - pairwise_distances( user_book_df.values,metric='cosine')
user_sim= 1 - pairwise_distances( user_book_df.values,metric='correlation')
user_sim

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)

#Set the index and column names to user ids 
user_sim_df.index = book_df.UserID.unique()
user_sim_df.columns = book_df.UserID.unique()

user_sim_df.iloc[0:50, 0:50]

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:50, 0:50]

user_sim_df.idxmax(axis=1)[0:50]

book_df[(book_df['UserID']==276813 ) | (book_df['UserID']==3546)]

user_1=book_df[book_df['UserID']==276872]
user_2=book_df[book_df['UserID']==161677]

user_1.BookTitle
user_2.BookTitle

pd.merge(user_1,user_2,on='BookTitle',how='outer')

#Alternative Method:
#(Recommending Similar Movies)

book_df.groupby('BookTitle')['BookRating'].mean().sort_values(ascending=False).head()

ratings = pd.DataFrame(book_df.groupby('BookTitle')['BookRating'].mean())

ratings.head()

ratings['num of ratings'] = pd.DataFrame(book_df.groupby('BookTitle')['BookRating'].count())
ratings.head()

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)

plt.figure(figsize=(10,4))
ratings['BookRating'].hist(bins=70)

sns.jointplot(x='BookRating',y='num of ratings',data=ratings,alpha=0.5)

moviemat =book_df.pivot_table(index='UserID',columns='BookTitle',values='BookRating')
moviemat.head

ratings.sort_values('num of ratings',ascending=False).head(10)

ratings.head(12)

stardust_user_ratings = moviemat['Stardust']
theamber_user_ratings = moviemat['The Amber Spyglass (His Dark Materials, Book 3)']
stardust_user_ratings.head()

similar_to_stardust = moviemat.corrwith(stardust_user_ratings)
similar_to_theamber= moviemat.corrwith(theamber_user_ratings)

corr_stardust = pd.DataFrame(similar_to_stardust,columns=['Correlation'])
corr_stardust.dropna(inplace=True)
corr_stardust

corr_stardust.sort_values('Correlation',ascending=False).head()

corr_stardust = corr_stardust.join(ratings['num of ratings'])
corr_stardust

corr_stardust[corr_stardust['num of ratings']>3].sort_values('Correlation',ascending=False).head()

corr_theamber = pd.DataFrame(similar_to_theamber,columns=['Correlation'])

corr_theamber.dropna(inplace=True)

corr_theamber = corr_theamber.join(ratings['num of ratings'])

corr_theamber[corr_theamber['num of ratings']>3].sort_values('Correlation',ascending=False).head()

##conclusion:
##    we can conclude that, both The Amber Spyglass (His Dark Materials, Book 3) and Stardust have same ratings.
## corr_theamber