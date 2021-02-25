####customer_order_form####
#Defective % do not vary by centre for all countries
#defective % vary by centre by centre for at least one country
cust <-  read.csv('E:\\Tej\\Assignments\\Asgnmnt\\Hypothesis\\Costomer+OrderForm.csv')
View(cust)

cust$Phillippines2[cust$Phillippines=='Error Free']='1'
cust$Indonesia2[cust$Indonesia=='Error Free']='1'
cust$Malta2[cust$Malta=='Error Free']='1'
cust$India2[cust$India=='Error Free']='1'

cust$Phillippines2[cust$Phillippines=='Defective']='0'
cust$Indonesia2[cust$Indonesia=='Defective']='0'
cust$Malta2[cust$Malta=='Defective']='0'
cust$India2[cust$India=='Defective']='0'

stacked <- stack(cust)
stacked
a <- table(stacked$ind,stacked$values)
chisq.test(a)
#p-value is 0.2721 >0.05 --> Fail to reject Ho
#Thus,Defective % do not vary by centre for all countries
