# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 22:42:07 2024

@author: Hector
"""

import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return np.cos(2*np.pi*t) * np.exp(-t) #Se define un oscilador amortiguado



fig = plt.figure(figsize=plt.figaspect(3.)) #Se crea una figura cuyo argumento a especificar sera el tamaño
                                            #El cual se calcula usando la funcion figaspect que requiere una relacion de aspecto
fig.suptitle('Dos subgráficas') #Establece el titulo superior

# Primera subgráfica
ax = fig.add_subplot(2, 1, 1) #Se crea el entorno para 2 graficas en vertical (2 filas, 1 columna, indice 1)

t1 = np.arange(0.0, 5.0, 0.1) 
t2 = np.arange(0.0, 5.0, 0.02)
t3 = np.arange(0.0, 2.0, 0.01) #No se usa en ningun lado del codigo

ax.plot(t1, f(t1), 'bo', t2, f(t2), 'k--', markerfacecolor='green') #Se usa t1 para muestrear la funcion con puntos discretos cada 0.1 unidades
                                                                    #Con t2 se hace un muestreo mas fino usando lineas punteadas cada 0.02 unidades
                                                                    #'bo' y k-- establecen los simbolos (cinculos, linea punteada), con los que se muestreara la funcion                
ax.grid(True) #Se activa la cuadricula 
ax.set_ylabel('Oscilador amortiguado') #Titulo en el eje Y

# Segunda subgráfica
ax = fig.add_subplot(2, 1, 2, projection='3d') #La 2da grafica se mostrara en el indice 2 (posicion baja del entorno creado anteriormente)
                                                #Dicha 2da grafica se mostrara como una proyeccion en 3d

X = np.arange(-30, 30, 0.25) #los limites en X,Y (primeras 2 entradas) definen el dominio en el cual sera evaluada la funcion
Y = np.arange(-30, 30, 0.25)
X, Y = np.meshgrid(X, Y) #Usando X,Y se crea la cuadricula XY, o sea el dominio de la funcion
R = np.sqrt(X**2 + Y**2) #Las variables R,Z definen la superficie 3D
Z = np.sin(R) #R calcula la distancia al origen y Z calcula la funcion seno de dicho punto, creando asi una superficie en forma de ondas

surf = ax.plot_surface(X, Y, Z, rstride=3, cstride=3,  #rstride=2 es el espaciado entre filas y cstride entre columnas
                       linewidth=0, antialiased=False) #De alguna manera define tambien la resolucion de la grafica al dibujar casa 2 filas y columnas
                                                        #Cambiando el calor a 1 aumenta la resolucion
                                                        #antialiased=False mejora el rendimiento al no hacer tan suave la funcion
ax.set_zlim(-4, 4) #Limite en el eje Z, si se le da un limite pequeño, la grafica se ve estirada a lo alto
#Un limite como 4 permite ver con mas claridad la grafica, con poca altura

plt.show()