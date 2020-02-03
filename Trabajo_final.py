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
from statsmodels.formula.api import ols
import MDA.graficas as gr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools

# Change working directory 
gr.chdir('/Users/Diego/Desktop/EDEM_DIEGO/02_CURSO/00_REPOSITORIOS/TRABAJO-GRUPO')

#Reads data from CSV file and stores it in a dataframe 
stud = pd.read_csv ("stud_per.csv", sep=',', decimal='.')
stud_ols = pd.read_csv ("stud_per_cat.csv", sep=',', decimal='.')

stud.shape
stud.head()
stud.tail()
#QC OK

stud_ols.shape
stud_ols.head()
stud_ols.tail()
#QC OK

# Section 1. Variable descriptives: Describe numerically / graphically
# the variables involved in your analyses. Begin always with your
# target variable


# Data cleaning
# We keep only GP students and those who have a final grade > 0
stud = stud[stud.school == 'GP']
stud = stud[stud.final_grade>0]

stud_ols = stud_ols[stud_ols.GP_school == 1]
stud_ols = stud_ols[stud_ols.final_grade>0]

# Graphics
#---------------------VARIABLE FINAL_GRADE---------------------------------
gr.histogr(stud.final_grade, nsteps = 10, xlabel = "Frequency", ylabel = "Final grade", title = "Fig. 1: Final grade", source = 'stud_per.csv', legend_position = 'top-left')

#---------------------VARIABLE FAILURES---------------------------------
gr.gbarras(stud.failures, stud.failures, xlabel = 'Variable: Failures', ylabel = 'Percentage', title = 'Fig. 2: Percentage of failures ', source = 'stud_per.csv')

#---------------------VARIABLE ABSENCES ---------------------------------
gr.histogr(stud.absences, xlabel = "Frequency", ylabel = "absences", title = "Fig. 3: Number of absences", source = 'stud_per.csv', legend_position = 'top-right')

#---------------------VARIABLE SEX---------------------------------
gr.gbarras(stud.sex, stud.sex, xlabel = 'Variable: Failures', ylabel = 'Percentage', title = 'Fig. 2: Percentage of failures ', source = 'stud_per.csv')

#---------------------VARIABLE MJOB---------------------------------
gr.gbarras(stud.Mjob, stud.Mjob, xlabel = 'Variable: Mjob', ylabel = 'Percentage', title = 'Fig x. Percentage of Mjob ', source = 'stud_per.csv')

#---------------------VARIABLE FJOB---------------------------------
gr.gbarras(stud.Fjob, stud.Fjob, xlabel = 'Variable: Fjob', ylabel = 'Percentage', title = 'Fig x. Percentage of Fjob ', source = 'stud_per.csv')

#---------------------VARIABLE MEDU---------------------------------
gr.gbarras(stud.Medu, stud.Medu, xlabel = 'Variable: Medu', ylabel = 'Percentage', title = 'Fig x. Percentage of Medu ', source = 'stud_per.csv')

#---------------------VARIABLE FEDU---------------------------------
gr.gbarras(stud.Fedu, stud.Fedu, xlabel = 'Variable: Fedu', ylabel = 'Percentage', title = 'Fig x. Percentage of Fedu ', source = 'stud_per.csv')

#---------------------VARIABLE STUDYTIME---------------------------------
gr.gbarras(stud.studytime, stud.studytime, xlabel = 'Variable: studytime', ylabel = 'Percentage', title = 'Fig. x: Percentage of studytime ', source = 'stud_per.csv')

#---------------------VARIABLE SCHOOLSUP---------------------------------
gr.gbarras(stud.schoolsup, stud.schoolsup, xlabel = 'Variable: schoolsup', ylabel = 'Percentage', title = 'Fig. x: Percentage of shoolsup', source = 'stud_per.csv', legend_position = 'top-right')

#---------------------VARIABLE GOOUT---------------------------------
gr.gbarras(stud.goout, stud.goout, xlabel = 'Variable: goout', ylabel = 'Percentage', title = 'Fig. x: Percentage of goout ', source = 'stud_per.csv', legend_position = 'top-left')

#---------------------VARIABLE HIGHER---------------------------------
gr.gbarras(stud.higher, stud.higher, xlabel = 'Variable: higher', ylabel = 'Percentage', title = 'Fig. x: Percentage of higher ', source = 'stud_per.csv', legend_position = 'top-left')

#---------------------VARIABLE FREETIME---------------------------------
gr.gbarras(stud.freetime, stud.freetime, xlabel = 'Variable: freetime', ylabel = 'Percentage', title = 'Fig. x: Percentage of freetime ', source = 'stud_per.csv', legend_position = 'top-left')

#---------------------VARIABLE DALC---------------------------------
gr.gbarras(stud.Dalc, stud.Dalc, xlabel = 'Variable: Dalc', ylabel = 'Percentage', title = 'Fig. x: Percentage of Dalc ', source = 'stud_per.csv', legend_position = 'top-right')

#---------------------VARIABLE WALC---------------------------------
gr.gbarras(stud.Walc, stud.Walc, xlabel = 'Variable: Walc', ylabel = 'Percentage', title = 'Fig. x: Percentage of Walc ', source = 'stud_per.csv', legend_position = 'top-right')


#Using Pearson Correlation
plt.figure(figsize=(36,32))
cor2 = stud_ols.corr()
sns.heatmap(cor2, vmin=-1, cmap='coolwarm')
plt.savefig('CorrelationMatrix.pdf')
plt.show()


# Buscamos las variables con una correlación significativa con final_grade (p-value<0.05)
cols=[]
for col in stud_ols.columns:
    if col != 'final_grade':
        model1 = ols('final_grade ~ ' + col,data = stud_ols).fit()
        summary = model1.summary2()
        if model1.pvalues[1]<0.05:
            cols.append(col)
        print(summary)
        print('P-value: ', model1.pvalues[1])


#combinations = combinations[1::]
#for i in combinations:
#    colnames = ''
#    for colname in i:
#        if colname != 'final_grade':
#            colnames = colnames + '+' + colname
#    model1 = ols('final_grade ~ ' + colnames,data = stud_ols).fit()
#    model1.summary2()

colsdef = model1.pvalues[1::]
colsdef = colsdef[colsdef<0.05]

colnames = ''
for colname in colsdef.index:
    colnames = colnames + '+' + colname
model1 = ols('final_grade ~ ' + colnames,data = stud_ols).fit()
model1.summary2()


model1 = ols('final_grade ~ failures + absences + health_Mjob + services_Mjob + teacher_Fjob + studytime_2 + yes_schoolsup + Walc_4',data = stud_ols).fit()
model1.summary2()






