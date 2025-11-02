# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:36:51 2024

@author: Hector
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pmgauss(x, y):
    r1 = (x-1)**2 + (y-2)**2
    r2 = (x-3)**2 + (y-1)**2
    return 2*np.exp(-0.5*r1) - 3*np.exp(-2*r2) #Define la funcion bi-valuada 


a, b = 8, 7 
x = np.linspace(0, a, 60)
y = np.linspace(0, b, 45)
X, Y = np.meshgrid(x, y) #Define el dominio de la funcion
Z = pmgauss(X, Y) #Se evalua dicho dominio y se define la superficie 3d

fig, ax = plt.subplots(1, 2, figsize=(9.2, 4),subplot_kw={'projection': '3d'})
for i in range(2):
    ax[i].set_zlim(-3, 3) #Limite en el eje z
    ax[i].xaxis.set_ticks(range(a+1)) # Se ponen los "ticks"
    ax[i].yaxis.set_ticks(range(b+1))
    ax[i].set_xlabel(r'$x$')
    ax[i].set_ylabel(r'$y$')
    ax[i].set_zlabel(r'$f(x,y)$')
    ax[i].view_init(40, -30)

# Hace dos gráficas

plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96,
top=0.96, wspace=0.05)
#Número máximo de muestras utilizadas en cada dirección. Si los datos de entrada son mayores, se reducirá su resolución (mediante segmentación) a este número de puntos.
p0 = ax[0].plot_wireframe(X, Y, Z, rcount=40, ccount=40,
color='C1')

p1 = ax[1].plot_surface(X, Y, Z, rcount=50, ccount=50,
color='C1')

plt.subplots_adjust(left=0.0)