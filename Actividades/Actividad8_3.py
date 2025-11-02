#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:51:23 2025

@author: isaias-gl
"""
import matplotlib.pyplot as plt
import pandas as pd

# Importamos los datos de un archivo txt
data = pd.read_csv("COBE.txt",header=0,delim_whitespace = True)
# Recopilamos los datos de las tres columnas
nu= data.iloc[:,0]
I= data.iloc[:,1]
sigma=data.iloc[:,2]

plt.figure(figsize=(10, 6))
plt.plot(nu, I, 'bo-')
plt.xlabel('Frecuencia')
plt.ylabel('Intensidad')
plt.title('Datos del COBE (Fondo cósmico de microondas)')
plt.grid(True)
plt.legend()
plt.show()


