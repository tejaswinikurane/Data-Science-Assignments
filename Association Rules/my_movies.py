# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:41:47 2020

@author: Dhotre
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules

#importig dataset
data=pd.read_csv('D:\\ML Docs\\Excelr\\Assignments\\Association rules\\my_movies.csv')
data.describe()

#removing unwanted columns
data_apr=data.iloc[:,5:]
data_apr

#getting frequent itemsets
frequent_itemsets1 =  apriori(data_apr,min_support = 0.05,max_len = 3,use_colnames = True)
frequent_itemsets=frequent_itemsets1.sort_values('support', ascending = False, inplace = False)

#barplot of top 10 items with highest support
plt.figure(figsize = (25,10))
plt.bar(x= list(range(0,11)), height =  frequent_itemsets.support.iloc[0:11,]);plt.xticks(list(range(0,11)),frequent_itemsets.itemsets[0:11])
plt.xlabel('itemsets');plt.ylabel('support values')

#getting association rules
rules1 = association_rules(frequent_itemsets, metric = 'lift', min_threshold=1)
rules= rules1.sort_values('lift', ascending = False, inplace = False)

#removing redundancy
def slist(i):
    return (sorted(list(i)))

concat1 = rules.antecedents.apply(slist)+ rules.consequents.apply(slist) 
concat = concat1.apply(sorted) #ascending order

rule_sets = list(concat) #converting series to list

uniq_rule_sets = [list(m) for m in set(tuple(i) for i in rule_sets)]

index_rules = []
for i in uniq_rule_sets:
    index_rules.append(rule_sets.index(i))
    
rules_no_red = rules.iloc[index_rules,:]
final_rules = rules_no_red.sort_values('lift',ascending = False , inplace = False)

#cutomers who watched 'sixth sense' and 'LOTR1' have also watched 'Green Mile'
#person watching ;Green Mile' most probably watched 'LOTR1' and 'Harry Potter'
#persons who watched 'Harry Potter 1' have watched 'LOTR1'
# and many more such rules can be made as per the mentioned in rules_no_red and giving offers or discounts to the audeience as per
#the formed rules will yield the better profits