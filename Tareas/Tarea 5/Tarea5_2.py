#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 18:39:55 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Datos proporcionados
E = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # MeV
f = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])  # MeV
sigma = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])  # MeV

# Crear spline cúbico usando scipy
# 'natural' significa que la segunda derivada en los extremos es cero
cs = CubicSpline(E, f, bc_type='natural')

# Puntos para graficar (alta resolución)
E_plot = np.linspace(0, 200, 500)
f_spline = cs(E_plot)

# Gráfica
plt.figure(figsize=(12, 8))

# Gráfica principal
plt.errorbar(E, f, yerr=sigma, fmt='o', label='Datos experimentales', 
             capsize=5, markersize=8, linewidth=2, color='blue', alpha=0.7)
plt.plot(E_plot, f_spline, 'r-', label='Spline Cúbico', linewidth=2.5)
plt.xlabel('Energía (MeV)', fontsize=14)
plt.ylabel('f(E) (MeV)', fontsize=14)
plt.title('Ajuste por Splines Cúbicos - Sección Eficaz de Resonancia', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 200)

# Estimación de Er y Γ
f_max = np.max(f_spline)
half_max = f_max / 2

# Encontrar donde la curva cruza la mitad del máximo
crossings = []
for i in range(len(f_spline) - 1):
    if (f_spline[i] - half_max) * (f_spline[i+1] - half_max) <= 0:
        # Interpolación lineal para encontrar el punto exacto
        x1, x2 = E_plot[i], E_plot[i+1]
        y1, y2 = f_spline[i], f_spline[i+1]
        x_cross = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        crossings.append(x_cross)

if len(crossings) >= 2:
    E_left, E_right = crossings[0], crossings[-1]
    Gamma = E_right - E_left
    Er = E_plot[np.argmax(f_spline)]
    
    # Marcar Er y Γ en la gráfica
    plt.axvline(x=Er, color='green', linestyle='--', alpha=0.7, label=f'$E_r$ = {Er:.1f} MeV')
    plt.axvline(x=E_left, color='orange', linestyle='--', alpha=0.7)
    plt.axvline(x=E_right, color='orange', linestyle='--', alpha=0.7, label=f'Γ = {Gamma:.1f} MeV')
    plt.axhline(y=half_max, color='purple', linestyle='--', alpha=0.7, label='Mitad del máximo')
    plt.axhline(y=f_max, color='red', linestyle='--', alpha=0.5, label='Máximo')
    
    # Resultados numéricos
    print("=== RESULTADOS SPLINES CÚBICOS ===")
    print(f"Energía de resonancia estimada Er = {Er:.2f} MeV")
    print(f"Ancho a la mitad del máximo Γ = {Gamma:.2f} MeV")
    print(f"Valores teóricos: Er = 78 MeV, Γ = 55 MeV")
    print(f"Diferencia en Er: {abs(Er-78):.2f} MeV")
    print(f"Diferencia en Γ: {abs(Gamma-55):.2f} MeV")
    
else:
    print("No se pudieron encontrar los puntos de cruce")

plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

