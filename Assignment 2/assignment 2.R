####SET+2####
#Q1-
pnorm(50,45,8)

#Q2 a) probability of employees older than 44
1-pnorm(44,38,6)  #=0.1586
#probability of employee =s between age 38-44
pnorm(44,38,6)-pnorm(38,38,6)  #=0.3413
# Thus A.	More employees at the processing center are older than 44 than between 38 and 44= False

#Q2 b) probability of employees younger than 30
pnorm(30,38,6)
 #Thus,no of emlployees = 0.091*400 =36.4
 # Thus,B.	A training program for employees under the age of 30 at the center would be expected to attract about 36 employees= TRUE

####SET4####
#Q3_probability that in any given week, there will be an investigation is
a <- pnorm(55,50,40)-pnorm(45,50,40)
bb <- 1-a