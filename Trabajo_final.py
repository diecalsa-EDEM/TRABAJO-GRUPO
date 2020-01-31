#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:23:36 2020

@author: germanvalera
"""

#Resets ALL (Careful This is a "magic" function then it doesn't run as script) 
#reset -f   

#load basiclibraries
import os
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype #For definition of custom categorical data types (ordinal if necesary)
import matplotlib.pyplot as plt
import seaborn as sns  # For hi level, Pandas oriented, graphics
import scipy.stats as stats  # For statistical inference 



os.getcwd()

# Change working directory
os.chdir('/Users/germanvalera/Edem/Fundamentos/Python')
os.getcwd()
#Reads data from CSV file and stores it in a dataframe called rentals_2011
# Pay atention to the specific format of your CSV data (; , or , .)
stud = pd.read_csv ("stud_per.csv", sep=',', decimal='.')
stud.shape
stud.head()
#QC OK


#Section 1. Variable descriptives: Describe numerically / graphically
#the variables involved in your analyses. Begin always with your
#target variable

#Numericas: Final_grade, ausencias, study_time
#Nominal: Romantic


#---------------------VARIABLE ROMANTIC---------------------------------
mytable = pd.crosstab(df.romantic, columns="count", normalize='columns')*100
print(mytable)
print (round(mytable,1))
plt.bar(mytable.index, mytable['count'])
plt.xlabel('Variable: Romantic')
plt.ylabel('Percentage')
plt.title('Figure x. Percentage of ROMANTIC ')



#---------------------VARIABLE FINAL_GRADE---------------------------------

new_stud = stud.drop([349])
res_target_describe = new_stud['final_grade'].describe()
print(res_target)
print(res_target_describe)

z = res_target.describe()
me = res_target_describe[1]
sde = res_target_describe[2]
ne = res_target_describe[4]


x = stud['final_grade']

plt.hist(x, bins=10, edgecolor='black')
plt.title('Figure1 . Histogram of students final grade')
plt.ylabel('Frecuency')
plt.xlabel('Final Grade')
plt.xticks(np.arange(0, 22, step=2))
plt.axvline(x=me,linewidth=1,linestyle= 'solid',color="red", label='Mean')
plt.axvline(x=me-sde,linewidth=1,linestyle= 'dashed',color="pink", label='- 1 S.D.')
plt.axvline(x=me + sde,linewidth=1,linestyle= 'dashed',color="pink", label='+ 1 S.D.')
plt.legend(loc='upper left', bbox_to_anchor=(0.73, 0.98))



#---------------------VARIABLE STUDYTIME---------------------------------

studytime_describe = new_stud['studytime'].describe()
print(studytime_describe)

z = studytime.describe()
me = studytime_describe[1]
sde = studytime_describe[2]
ne = studytime_describe[4]

#Cambio a nominal

stud.studytime = stud.studytime.replace(to_replace="1 Hora", value="1 Hora")
stud.studytime = stud.studytime.replace(to_replace="2 Hora", value="2 Horas")
stud.studytime = stud.studytime.replace(to_replace="3 Hora", value="3 Horas")
stud.studytime = stud.studytime.replace(to_replace="4 Hora", value="4 Horas")

studytime_describe = new_stud['studytime'].describe()

mytable = pd.crosstab(stud.studytime, columns="count", normalize='columns')*100
print(mytable)
print (round(mytable,1))
plt.bar(mytable.index, mytable['count'])
plt.xlabel('Variable: STUDYTIME')
plt.ylabel('Percentage')
plt.title('Figure x. Percentage of STUDYTIME ')




#---------------------VARIABLE INTERNET ACCESS ---------------------------------



res_internet = stud['internet'].describe()
print(res_internet)


mytable_internet = pd.crosstab(index=stud["internet"], columns="count")
print(mytable_internet)

n = mytable_internet.sum()

mytable_internet2 = (mytable_internet/n)*100 
mytable_internet2 = (round(mytable_internet2,1))

print(mytable_internet2)
del (mytable_internet)

plt.bar(mytable_internet2.index, mytable_internet2['count'])
plt.xlabel('Internet Access')
plt.ylabel('Number Students ')
plt.title('Figure X. Percentage of Students with internet access')



#---------------------VARIABLE ABSENCES ---------------------------------

res_absences = stud['absences'].describe()
print(res_absences)


z = res_absences.describe()
me = res_absences[1]
sde = res_absences[2]
ne = res_absences[4]


### Recode cnt to string
stud.loc[  (stud['absences']<(me-sde)) ,"Absences_str"]= "Low absences"
stud.loc[ ((stud['absences']>=(me-sde)) & (stud['absences']<=(me+sde))) ,"Absences_str"]= "Average absences"
stud.loc[  (stud['absences']>(me+sde)) ,"Absences_str"]= "High absences"



#frequencies & barchart
mytable = pd.crosstab(stud.Absences_str, columns="count", normalize='columns')*100
print(mytable)
print (round(mytable,1))
plt.bar(mytable.index, mytable['count'])


