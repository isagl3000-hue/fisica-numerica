#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:09:43 2025

@author: isaias-gl
"""

from matplotlib.pyplot import *
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

N=1000 
         # Número de pasos
x0=1           # Posición inicial
v0=0          # Velocidad inicial
tau=150.0 
tau2=150.0         # Tiempo en segundos de la simulación
h=tau/float(N-1)
h2=tau2/float(N-1)
m=0.5
k=2
b=0.1
b2=0.05
omega=0.5
F0=0.2

# Generamos un arreglo de Nx2 para almacenar posición y velocidad
y=zeros([N,2])
# tomamos los valores del estado inicial
y[0,0]=x0
y[0,1]=v0

# Generamos tiempos igualmente espaciados
tiempo=linspace(0,tau,N)

def EDO(estado,tiempo):
    f0=estado[1]
    f1=-(k/m)*estado[0]-(b/m)*estado[1]
    return array([f0,f1])



def Euler(y,t,h,f): 
    y_p=y+h*f(y,t)  # Calculamos el valor siguiente de y predictor
    y_c=y+h*(f(y,t)+f(y_p,t+h))/2.0  # Calculamos el valor siguiente de y corregido
    return y_c

# Ahora calculamos!
for j in range(N-1):
    y[j+1]=Euler(y[j],tiempo[j],h,EDO)
                 
    
# Ahora graficamos
xdatos=[y[j,0] for j in range(N)]
vdatos=[y[j,1] for j in range(N)]


fig = plt.figure(figsize=plt.figaspect(4.))
fig.suptitle('tres subgráficas') 
ax = fig.add_subplot(3, 2, 1) 
ax.plot(tiempo, xdatos, '-r')   
ax = fig.add_subplot(3, 2, 2)                                                                   #'bo' y k-- establecen los simbolos (cinculos, linea punteada), con los que se muestreara la funcion
ax.plot(tiempo, vdatos, '-b')
ax = fig.add_subplot(3, 2, 3)                                                                   #'bo' y k-- establecen los simbolos (cinculos, linea punteada), con los que se muestreara la funcion
ax.plot(xdatos, vdatos, '-g')
#plot(tiempo,xdatos,'-r')
#plot(tiempo,vdatos,'-b')
#plot(xdatos,vdatos,'-r')
xlabel('Tiempo')
ylabel('Posición y velocidad')
show()





