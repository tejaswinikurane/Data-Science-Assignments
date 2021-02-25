

# -*- coding: utf-8 -*-
"""Created on Sat Sep 26 15:31:23 2020

@author: Admin"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

books= pd.read_csv('D:\\ML Docs\\Excelr\\Assignments\\Association rules\\book.csv')
books

frequent_itemsets= apriori(books,min_support=0.05,use_colnames=True,max_len=3)
frequent_itemsets

frequent_itemsets=frequent_itemsets.sort_values('support',ascending=False,inplace= False)

plt.figure(figsize=(25,10))
plt.bar(x=list(range(1,11)),height=frequent_itemsets.support[1:11]);plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('itemsets');plt.ylabel('support')

rules= association_rules(frequent_itemsets,metric="lift",min_threshold=1)
rules1=rules.sort_values('lift',ascending=False,inplace=False)

#removing redundancy
def slist(i):
    return (sorted(list(i))) #sort - to sort the string alphabetically

concat=  rules1.antecedents.apply(slist) + rules1.consequents.apply(slist)
concat=concat.apply(sorted)  #sort - to sort the string alphabetically

rule_sets=list(concat) #converting concat to list from series

uni_rule_sets= [list(m) for m in set(tuple(i) for i in rule_sets )] #set- to remove the duplicate elements

index_sets= []
for i in uni_rule_sets:
    index_sets.append(rule_sets.index(i))
    
# getting rules without any redudancy 
rules_no_red = rules.iloc[index_sets,:]

#sorting rules wrt lift associated with them
rules_no_red.sort_values('lift',ascending= False).head()

#persons who bought 'Italcook' also bought 'CookBks' and 'ArtBks'
#persons buying 'GeogBks' and 'ChildBks' have bought 'Italcook'
#perosns buying 'CookBks' also buy 'ItalBks' 
# and many more such rules can be made as per the mentioned in rules_no_red and giving offers or discounts to the audeience as per
#the formed rules will yield the better profits