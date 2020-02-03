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
gr.histogr(stud.final_grade, nsteps = 10, xlabel = "Frequency", ylabel = "Final grade", title = "Fig. 1: Final grade", source = 'stud_per.csv', legend_position = 'top-left')

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


combinations = combinations[1::]
for i in combinations:
    colnames = ''
    for colname in i:
        if colname != 'final_grade':
            colnames = colnames + '+' + colname
    model1 = ols('final_grade ~ ' + colnames,data = stud_ols).fit()
    model1.summary2()

a = model1.pvalues[1::]
counter = 0
cols2 = cols
for i in model1.pvalues[1::]:
    if i > 0.05:
        cols2.remove(a.index[counter])
    counter+=1

colnames = ''
for colname in cols:
    colnames = colnames + '+' + colname
model1 = ols('final_grade ~ ' + colnames,data = stud_ols).fit()
model1.summary2()


model1 = ols('final_grade ~ failures + absences + F_sex + services_Mjob + teacher_Fjob + studytime_2 + yes_schoolsup + goout_2 + Walc_4',data = stud_ols).fit()
model1.summary2()

combinations = []
for L in range(0, len(cols)+1):
    for subset in itertools.combinations(cols, L):
        combinations.append(subset)
        print(subset)

#        cols.remove(i)
        
#from sklearn.preprocessing import MinMaxScaler
#
#scaler = MinMaxScaler(feature_range=[-1, 1]) 
#stud_ols_scaled = scaler.fit_transform(stud_ols)
#
##Ajustando el PCA con nuestros datos
#pca = PCA().fit(stud_ols_scaled)
##Gráfica de la suma acumulativa de la varianza explicada
#plt.figure()
#plt.grid()
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Número de Componentes')
#plt.xticks(np.arange(0,80,4))
#plt.yticks(np.arange(0,1,0.05))
#plt.ylabel('Varianza (%)') #para cada componente
#plt.title('Student final grades Dataset Explained Variance')
#plt.show()

