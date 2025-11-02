# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:10:51 2025

@author: IsaiasGL
"""
import numpy as np
import matplotlib.pyplot as plot

t = np.linspace(1.5,20.5,6000) #Defninimos el dominio de la funcion

x = t-1.6*np.cos(24*t) #Definimos las funciones parametrizadas
y = t-1.6*np.cos(25*t) 

#Graficamos
plot.figure(figsize=(7,7))
plot.plot(x,y)
plot.title("Curva paramétrica")
plot.xlabel("x(t)")
plot.ylabel("y(t)")
plot.axis('equal')
plot.grid(True)
plot.tight_layout()

plot.show()


