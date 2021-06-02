# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:39:03 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import tweepy
from wordcloud import WordCloud
from textblob import TextBlob

#authentication
consumer_key = "e5er9Ba7ACWjxmkCSCEpdMTv2"
consumer_secret = "Qv1BfqtglOWoh3F7olX1G0Tsa2JbDuus7KEdSmJkGL1JMKpQwT"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)

#getting first 200 tweets
userID = 'realDonaldTrump'
tweets = api.user_timeline(screen_name = userID,
                           count = 200,
                           include_rts = False,
                           tweet_mode   = 'extended'
                           )

oldest = tweets[-1].id - 1

all_tweets = []
all_tweets.extend(tweets)
oldest_id = tweets[-1].id
while True:
    tweets = api.user_timeline(screen_name = userID,
                           count = 200,
                           include_rts = False,
                           max_id = oldest_id - 1,
                           tweet_mode   = 'extended'
                           )
    if len(tweets) == 0:
        break
    oldest_id = tweets[-1].id
    all_tweets.extend(tweets)
    print('no. of tweets downloaded : {}'.format(len(all_tweets)))

#data cleaning
from pandas import DataFrame
outtweets = [[tweet.id_str, 
              tweet.created_at, 
              tweet.favorite_count, 
              tweet.retweet_count,
              tweet.full_text.encode("utf-8").decode("utf-8")]
             for idx,tweet in enumerate(all_tweets)]
df = DataFrame(outtweets,columns=["id","created_at","favorite_count","retweet_count", "text"])

import re
def cleanUpTweet(txt):
    # Remove mentions
    txt = re.sub(r'@[A-Za-z0-9_]+', '', txt)
    # Remove hashtags
    txt = re.sub(r'#', '', txt)
    # Remove retweets:
    txt = re.sub(r'RT : ', '', txt)
    # Remove urls
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', txt)
    txt.lower
    return txt

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(df.text)
sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

#frequency plot
frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'blue')
plt.title("Most Frequently Occuring Words - Top 30")
#https, bjp, india, people ,modi

#plotting wordcloud
all_words = ' '.join([text for text in df['text']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.title('Dhruv Rathee Tweet Analysis')
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
