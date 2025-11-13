#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:02:40 2025

@author: isaias-gl
"""

import random
import numpy as np
import matplotlib.pyplot as plt

tamaños=[10000,20000,50000]
datos_aleatorios={}

for n in tamaños:
    datos_aleatorios[n]=[random.random() for _ in range(n)]
    print(f"Generados {n} numeros aleatorios")
    
op_bins=[10,50,100,200]

fig, axes = plt.subplots(len(tamaños), len(op_bins), figsize=(20,15))

for i,n in enumerate(tamaños):
    for j, bins in enumerate(op_bins):
        ax=axes[i,j]
        ax.hist(datos_aleatorios[n], bins=bins, density=True, alpha=0.7, color="blue", edgecolor="black")
        ax.set_title(f"n={n:,}, bins={bins}")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidad")
        ax.grid(True, alpha=0.3)
        
plt.tight_layout()
plt.suptitle('Distribución de números aleatorios uniformes [0,1) con diferentes bins', 
             fontsize=16, y=1.02)
plt.show()

print("\n"+"="*50)
print("Mientras mayores son los Bins, menos uniforme se ve la distribución, se aprecian mas detalles de la generacion de numeros aleatorios")


        
        