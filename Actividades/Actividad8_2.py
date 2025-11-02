#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:39:12 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 2(a) Guardar datos en arreglos
# ----------------------------

i = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
E = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # MeV
f = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])  # MeV
sigma = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])  # MeV

# ----------------------------
# 2(b) Gráfica de f(Ei) vs Ei
# ----------------------------

plt.figure(figsize=(10, 6))
plt.plot(E, f, 'bo-', label='f(E) datos')
plt.xlabel('Energía E (MeV)')
plt.ylabel('f(E) (MeV)')
plt.title('Sección eficaz f(E) vs Energía')
plt.grid(True)
plt.legend()
plt.show()

# ----------------------------
# 2(c) Gráfica con barras de error
# ----------------------------

plt.figure(figsize=(10, 6))
plt.errorbar(E, f, yerr=sigma, fmt='ro', capsize=5, label='f(E) con error')
plt.xlabel('Energía E (MeV)')
plt.ylabel('f(E) (MeV)')
plt.title('Sección eficaz f(E) con incertidumbre σ')
plt.grid(True)
plt.legend()
plt.show()

# ----------------------------
# 2(d) Ajuste de Breit-Wigner
# ----------------------------

def breit_wigner(E, Er, Gamma):
    """Fórmula de Breit-Wigner para la sección eficaz."""
    return (1 / ((E - Er)**2 + (Gamma/2)**2))

# Parámetros dados
Er = 78  # MeV
Gamma = 55  # MeV

# Normalizamos Breit-Wigner para que coincida visualmente con los datos
E_continuous = np.linspace(0, 200, 300)
BW_curve = breit_wigner(E_continuous, Er, Gamma)
# Escalamos para que pase cerca de los datos (ajuste visual)
scale = f.max() / BW_curve.max()
BW_curve *= scale

plt.figure(figsize=(10, 6))
plt.errorbar(E, f, yerr=sigma, fmt='bo', capsize=5, label='Datos experimentales')
plt.plot(E_continuous, BW_curve, 'r-', label=f'Breit-Wigner (Er={Er}, Γ={Gamma})')
plt.xlabel('Energía E (MeV)')
plt.ylabel('f(E) (MeV)')
plt.title('Ajuste de Breit-Wigner a los datos')
plt.grid(True)
plt.legend()
plt.show()