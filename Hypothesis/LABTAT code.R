library(readxl)
LAB<-read_excel("E:/Tej/Assignments/Asgnmnt/Hypothesis/LabTAT.xls")   # ContractRenewal_Data(unstacked).xlsx
View(LAB)
Stacked_Data <- stack(LAB)
View(Stacked_Data)
attach(Stacked_Data)


shapiro.test(LAB$`Laboratory 1`)
shapiro.test(LAB$`Laboratory 2`)
shapiro.test(LAB$`Laboratory 3`)
shapiro.test(LAB$`Laboratory 4`)

summary(LAB)
hist(LAB$`Laboratory 1`)
hist(LAB$`Laboratory 2`)
hist(LAB$`Laboratory 3`)
hist(LAB$`Laboratory 4`)
qqnorm(LAB$`Laboratory 1`)

# Data is normally distributed
library(car)
# Test for vaiance
leveneTest(values ~ ind, data = Stacked_Data)
?leveneTest
Anova_results <- aov(values~ind,data = Stacked_Data)
summary(Anova_results)
print(Anova_results)
# p-value = 0.104 > 0.05 accept null hypothesis 
# All Proportions all equal 