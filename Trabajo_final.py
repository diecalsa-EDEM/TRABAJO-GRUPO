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

gr.chdir('/Users/Diego/Desktop/EDEM_DIEGO/02_CURSO/00_REPOSITORIOS/TRABAJO-GRUPO')

#Reads data from CSV file and stores it in a dataframe called rentals_2011
# Pay atention to the specific format of your CSV data (; , or , .)
stud = pd.read_csv ("stud_per.csv", sep=',', decimal='.')
stud_ols = pd.read_csv ("stud_per_cat.csv", sep=',', decimal='.')
stud.shape
stud.head()
#QC OK

#Section 1. Variable descriptives: Describe numerically / graphically
#the variables involved in your analyses. Begin always with your
#target variable

#Numericas: Final_grade, ausencias, study_time
#Nominal: Romantic

# Data cleaning
stud = stud[stud.school == 'GP']
stud = stud[stud.final_grade>0]

stud_ols = stud_ols[stud_ols.GP_school == 1]
stud_ols = stud_ols[stud_ols.final_grade>0]

#---------------------VARIABLE FINAL_GRADE---------------------------------
gr.histogr(stud.final_grade, nsteps = 20, xlabel = "Frequency", ylabel = "Final grade", title = "Fig. 1: Final grade", source = 'stud_per.csv', legend_position = 'top-left')

#---------------------VARIABLE ROMANTIC---------------------------------
gr.gbarras(stud.romantic, stud.romantic, xlabel = 'Variable: Romantic', ylabel = 'Percentage', title = 'Figure x. Percentage of ROMANTIC ', source = 'stud_per.csv')

#---------------------VARIABLE STUDYTIME---------------------------------
gr.gbarras(stud.studytime, stud.studytime, xlabel = 'Variable: studytime', ylabel = 'Percentage', title = 'Figure x. Percentage of studytime ', source = 'stud_per.csv')

#---------------------VARIABLE INTERNET ACCESS ---------------------------------
gr.gbarras(stud.internet, stud.internet, xlabel = 'Variable: internet', ylabel = 'Percentage', title = 'Figure x. Percentage of internet access ', source = 'stud_per.csv', legend_position = 'top-left')

#---------------------VARIABLE ABSENCES ---------------------------------
gr.histogr(stud.absences, xlabel = "Frequency", ylabel = "absences", title = "Fig. X: Number of absences", source = 'stud_per.csv', legend_position = 'top-right')


#Using Pearson Correlation
plt.figure(figsize=(36,32))
cor2 = stud_ols.corr()
sns.heatmap(cor2, vmin=-1, cmap='coolwarm')
plt.savefig('CorrelationMatrix.pdf')
plt.show()

model1 = ols('final_grade ~ failures + absences + M_sex + at_home_Mjob + health_Mjob + other_Mjob + services_Mjob + at_home_Fjob + health_Fjob + other_Fjob + services_Fjob',data = stud_ols).fit()
model1.summary2()

for col in stud_ols.columns:
    model1 = ols('final_grade ~ ' + col,data = stud_ols).fit()
    print(model1.summary2())
