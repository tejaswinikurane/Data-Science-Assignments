# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:07:22 2020

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup as bs
import re

reviews = []
for i in range(1,20):
    ip=[]
    url="https://www.amazon.in/Sony-HT-S20R-Soundbar-Bluetooth-Connectivity/dp/B084685MT1/ref=sr_1_1?crid=1AS0CVH00ZXCK&dchild=1&keywords=sony+home+theatre+5.1+with+bluetooth+with+bass&qid=1602513091&sprefix=sony+%2Caps%2C443&sr=8-1"+str(i)
    response = requests.get(url)
    soup=bs(response.content,"html.parser")
    reviews1=soup.findAll("span",attrs={"class","a-size-base review-text"})
    for i in range(len(reviews1)):
      ip.append(reviews1[i].text)    
    reviews=reviews+ip   
    
with open("ecom_rev.txt","w",encoding='utf8') as output:
    for i in reviews:
        output.write(i+"\n\n")
rev_string = " ".join(reviews)
rev_string = re.sub("[^A-Za-z" "]+"," ",rev_string).lower()
rev_string = re.sub("[0-9" "]+"," ",rev_string)
reviews_words = rev_string.split(" ")

#getting word cloud for reviews
stop_words = stopwords.words('english')
with open("stop.txt","r") as sw:
    stopwords = sw.read()
stopwords = stopwords.split("\n")
stp_wrds_final = stopwords+stop_words
reviews_words = [w for w in reviews_words if not w in stp_wrds_final]
rev_string = " ".join(reviews_words)#for word cloud
wordcloud_rev = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(rev_string)

plt.imshow(wordcloud_rev)

#positive words wordcloud
with open("positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
poswords = poswords[36:]
pos_rev = " ".join ([w for w in reviews_words if w in poswords])
wordcloud_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(pos_rev)

plt.imshow(wordcloud_pos)

#negative words wordcloud
with open("negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")
negwords = negwords[37:]
neg_rev = " ".join ([w for w in reviews_words if w in negwords])

wordcloud_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(neg_rev)

plt.imshow(wordcloud_neg)
