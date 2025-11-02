# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:33:35 2025

@author: IsaiasGL
"""

from numpy import *
from matplotlib.pyplot import *

xmin = -3
xmax = 5
N = 10000
ancho = (xmax - xmin)/N

def f(x):  #Defninimos la funcion a integrar
    return 1-x-4*x**3+3*x**5

def simpson(f,xmax,xmin,N):
    x = linspace(xmin, xmax, N)
    y = f(x)
    
    s = y[0]+y[-1]
    s += 4*sum(y[1:-1:2]) #indices pares
    s += 2*sum(y[2:-2:2]) #indices impares
    
    return (ancho/3)*s
print(simpson(f, xmax, xmin, N))



def trapesio(f,xmax,xmin,N):
    x = linspace(xmin, xmax, N)
    y = f(x)
    
    s = y[0]/2+y[-1]/2
    s += sum(y[1:-1:1]) #indices pares
    
    return (ancho)*s

print(trapesio(f, xmax, xmin, N))



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
