#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 22:02:58 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt

def lagrange(x_points, y_points, x):
    n = len(x_points)
    result = 0.0
    
    for i in range(n):
        # Calcular el polinomio base L_i(x)
        L_i = 1.0
        for j in range(n):
            if i != j:
                L_i *= (x - x_points[j]) / (x_points[i] - x_points[j])
        
        # Sumar el término y_i * L_i(x)
        result += y_points[i] * L_i
    
    return result

# Datos proporcionados
E = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # MeV
f = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])  # MeV
sigma = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])  # MeV

# Puntos para graficar (cada 5 MeV)
E_plot = np.arange(0, 201, 5)
f_plot = np.array([lagrange(E, f, x) for x in E_plot])

# Gráfica
plt.figure(figsize=(10, 6))
plt.errorbar(E, f, yerr=sigma, fmt='o', label='Datos experimentales', 
             capsize=5, markersize=6, linewidth=2)
plt.plot(E_plot, f_plot, 'r-', label='Interpolación de Lagrange', linewidth=2)
plt.xlabel('Energía (MeV)', fontsize=12)
plt.ylabel('f(E) (MeV)', fontsize=12)
plt.title('Ajuste de Lagrange para la Sección Eficaz', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Estimación de Er y Γ (Energía de resonancia y ancho)
f_max = np.max(f_plot)
half_max = f_max / 2

# Encontrar donde la curva cruza la mitad del máximo
crossings = []
for i in range(len(f_plot) - 1):
    if (f_plot[i] - half_max) * (f_plot[i+1] - half_max) <= 0:
        # Interpolación lineal para encontrar el punto exacto
        x1, x2 = E_plot[i], E_plot[i+1]
        y1, y2 = f_plot[i], f_plot[i+1]
        x_cross = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        crossings.append(x_cross)

if len(crossings) >= 2:
    E_left, E_right = crossings[0], crossings[-1]
    Gamma = E_right - E_left
    Er = E_plot[np.argmax(f_plot)]
    
    print("=== RESULTADOS ===")
    print(f"Energía de resonancia estimada Er = {Er:.2f} MeV")
    print(f"Ancho a la mitad del máximo Γ = {Gamma:.2f} MeV")
    print(f"Valores teóricos: Er = 78 MeV, Γ = 55 MeV")
    print(f"Diferencia en Er: {abs(Er-78):.2f} MeV")
    print(f"Diferencia en Γ: {abs(Gamma-55):.2f} MeV")
else:
    print("No se pudieron encontrar los puntos de cruce")
    
    
    
    
