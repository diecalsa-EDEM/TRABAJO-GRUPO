#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:52:44 2019

@author: Diego
"""

######### LIBRERÍAS ############
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
################################


######### FUNCIONES ############
def chdir(directory = ''):
    print(os.getcwd())
    os.chdir(directory)
    print(os.getcwd())
    
def histogr(dataset, nsteps = 4, xlabel = '', ylabel = '', title = '', source = '', legend_position = "top-right"):
    #Histograma
    #Mostrar datos estadísticos por pantalla
    data = dataset.describe()
    print(data)
    #Guardamos datos estadísticos más representativos
    max_dataset = data['max']
    min_dataset = data['min']
    n = data['count']
    m = data['mean']
    std = data['std']
    
    #Setup xticks rounded by its magnitude
    step = (max_dataset-min_dataset)/(nsteps)
    if step < 1:
        rfactor = 1
    elif step < 10:
        rfactor = 0  
    elif step < 100:
        rfactor = -1   
    elif step < 1000:
        rfactor = -2
    else:
        rfactor = -3    
    step = round(step,rfactor)
    
    #Mostrar histograma
    plt.hist(dataset,edgecolor = 'black', linewidth = 1)
    xticks = np.arange(round(min_dataset,rfactor),round(max_dataset+step,rfactor),step=step)
    #xticks
    plt.xticks(xticks.round(decimals = rfactor))
    #xlabel
    plt.xlabel(xlabel)
    #ylabel
    plt.ylabel(ylabel)
    #title
    plt.title(title)
    #Median and std lines
    plt.axvline(x = m, linewidth = 1, linestyle = 'solid', color = 'red', label = 'Mean')
    plt.axvline(x = m-std, linewidth = 1, linestyle = 'dashed', color = 'violet', label = '- 1 S.D.')
    plt.axvline(x = m+std, linewidth = 1, linestyle = 'dashed', color = 'violet', label = '+ 1 S.D.')
    #Legend
    plt.legend()
    #Calculate legend position
    xmin, xmax, ymin, ymax = plt.axis()
    if legend_position == "top-right":
        x = xmax - (xmax-xmin)*0.235
        y = ymax*0.5
    elif legend_position == "top-left":
        x = xmin + (xmax-xmin)*0.024
        y = ymax*0.785
    elif legend_position == "bottom-right":
        x = xmax - (xmax-xmin)*0.235
        y = ymin + (ymax-ymin)*0.04
    elif legend_position == "bottom-left":
        x = xmin + (xmax-xmin)*0.024
        y = ymin + (ymax-ymin)*0.04
    else:
        x = xmax - (xmax-xmin)*0.235
        y = ymax*0.5
    textstr = '$\mathrm{n}:%.0f$\n$\mathrm{Mean}:%.2f$\n$\mathrm{Std}:%.2f$'%(n,m,std)
    props = dict(boxstyle = 'round', facecolor = 'white', lw = 0.5)
    plt.text(x,y,textstr, bbox=props)
    #Source text
    plt.text(min_dataset-step*0.5,-ymax*0.25,'$\t{Source}:$' + source)
    plt.show()

def gbarras(index, values, *objects, aggfunc = 'count', xlabel = '', ylabel = '', title = '', porcentaje = True, ndecimals = 0, source = '', legend_position = "top-right"):
    #Obtener valores estadísticos importantes
    data = values.describe()
    n = data['count']
    
    #Seleccionar conteo o media para mostrar en gráfico
    if aggfunc == 'count':
        cname = 'count'
    elif aggfunc == 'mean':
        cname = 'mean'
    else:
        cname = 'count'
    
    #Generación de crosstab
    mytable = pd.crosstab(index = index,columns=cname,values = values,aggfunc = aggfunc) 
    
    #Mostrar datos en porcentaje
    if porcentaje:
        n = mytable.sum()
        mytable = (mytable/n)*100
    
    #Mostrar por pantalla la tabla
    print(mytable)
    
    #Generación de gráfico de resultados
    bar = plt.bar(mytable.index,mytable.iloc[:,0], edgecolor = 'black')
    
    #Actualizar ylabel
    if((porcentaje) & (ylabel == '')):
        plt.ylabel("%")
    elif((ylabel == '')):
        plt.ylabel("Frequency")
    else:
        plt.ylabel(ylabel)   
    #Actualizar xlabel
    plt.xlabel(xlabel)
    #Actualizar título
    plt.title(title)
    #Plot de dato / % en medio de la barra
    for rect in bar:
        height = rect.get_height()
        if porcentaje:
            if ndecimals==0:
                plt.text(rect.get_x() + rect.get_width()/2.0, height/2, str(int(round(height,ndecimals))) + " %", ha='center')
            else:
                plt.text(rect.get_x() + rect.get_width()/2.0, height/2, str(round(height,ndecimals)) + " %", ha='center')
        else:
            plt.text(rect.get_x() + rect.get_width()/2.0, height/2, str(int(round(height,ndecimals))), ha='center')
    #Obtener rango de ejes x e y
    xmin, xmax, ymin, ymax = plt.axis()
    #Actualizar texto de fuente
    plt.text(-0.5,-ymax*0.25,'$\t{Source}:$' + source)
    
    #Leyenda
    textstr = '$\mathrm{n}:%.0f$'%(n)
    props = dict(boxstyle = 'round', facecolor = 'white', lw = 0.5)
    if legend_position == "top-right":
        x = xmax*0.8
        y = ymax*0.9
    elif legend_position == "top-left":
        x = xmin + (xmax-xmin)*0.05
        y = ymax*0.9
    elif legend_position == "bottom-right":
        x = xmax*0.8
        y = ymin + (ymax-ymin)*0.1
    elif legend_position == "bottom-left":
        x = xmin + (xmax-xmin)*0.05
        y = ymin + (ymax-ymin)*0.1
    else:
        x = xmax*0.8
        y = ymax*0.9
        print("\nUnknown legend position. Try with one of theese options:\n1. ""top-right""\n2. ""top-left""\n3. ""bottom-right""\n4. ""bottom-left""")
    plt.text(x,y,textstr, bbox=props)
    #Mostrar gráfico
    plt.show()
    
################################