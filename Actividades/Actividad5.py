#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 11:10:50 2025

@author: isaias-gl
"""
import numpy as np

#PUNTO 1 =====================================================================
A = np.array([[1,1,1],[2,1,2],[3,2,4]]) #Creamos la matriz
print(A)
B = np.dot(A,A) #Calculamos el cuadrado
B1 = np.linalg.matrix_power(A, 2) #Otro metodo
print(B)
C = np.linalg.inv(A) #Calculamos la inversa
print(C)
D = np.dot(A,C) #COmprobamos que sea la inversa
print(D)

E = np.linalg.eig(A) #Calculamos valores y vectores propios 
print(E)

#PUNTO 2 =====================================================================

F = np.array([[2j,-1+1j],[1+1j,3j]])
G = np.dot(F,F)
print(G)
H = np.linalg.inv(F)
print(H)
I = np.dot(H,F)
print(I)

J = np.linalg.eig(F)
print("\n",J)

K = np.linalg.eigvals(F)
print(K)







