# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:12:35 2020

@author: Vinayak Dhotre
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tweepy
from wordcloud import WordCloud
#conda install -c conda-forge wordcloud
#credentials
consumer_key = "e5er9Ba7ACWjxmkCSCEpdMTv2"
consumer_secret = "Qv1BfqtglOWoh3F7olX1G0Tsa2JbDuus7KEdSmJkGL1JMKpQwT"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)

#getting first 200 tweets
userID= 'dhruv_rathee'
tweets = api.user_timeline(screen_name=userID, 
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended'
                           )

oldest = tweets[-1].id - 1

    
#info on fist 3tweets
for info in tweets[:3]:
     print("ID: {}".format(info.id))
     print(info.created_at)
     print("\n")

# =============================================================================
# =============================================================================
# ID: 1323694224957734914
# 2020-11-03 18:31:10
# Eagerly waiting to say bye bye to Doland Trump Chacha tomorrow 🤞
# 
# 
# ID: 1323226147572908040
# 2020-11-02 11:31:12
# Oh acha, aisa hai kya?
# 
# Are you talking about this video? Guys, can you please watch and share this video to know if she’s talking about this video or not?
# 
# 👉https://t.co/sDaPWuP8rP
# 
# Because I didn’t receive any payment https://t.co/Reu72RUQ3V
# 
# 
# ID: 1322927160340975617
# 2020-11-01 15:43:08
# New Video on France Attacks 
# 
# Watch: https://t.co/RKw8i3PAmI https://t.co/sF73vJtxVR
# =============================================================================
# =============================================================================

#extracting all tweets
all_tweets = []
all_tweets.extend(tweets)
oldest_id = tweets[-1].id
while True:
    tweets = api.user_timeline(screen_name=userID, 
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                           max_id = oldest_id - 1,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended'
                           )
    if len(tweets) == 0:
        break
    oldest_id = tweets[-1].id
    all_tweets.extend(tweets)
    print('N of tweets downloaded till now {}'.format(len(all_tweets))) 
    


#data cleaning
from pandas import DataFrame
outtweets = [[tweet.id_str, 
              tweet.created_at, 
              tweet.favorite_count, 
              tweet.retweet_count,
              tweet.full_text.encode("utf-8").decode("utf-8")]
             for idx,tweet in enumerate(all_tweets)]
df = DataFrame(outtweets,columns=["id","created_at","favorite_count","retweet_count", "text"])

#getting frequency of words using count vectorizer
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

