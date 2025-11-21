#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 20:53:09 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime

# Datos proporcionados
E = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # MeV
f = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])  # MeV
sigma = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])  # MeV

# Función de Breit-Wigner
def breit_wigner(x, a1, a2, a3):
    """
    a1 = f_r (altura máxima)
    a2 = E_r (energía de resonancia) 
    a3 = Γ²/4
    """
    return a1 / ((x - a2)**2 + a3)

# Función chi-cuadrado a minimizar
def chi_square(params, x, y, errors):
    a1, a2, a3 = params
    y_pred = breit_wigner(x, a1, a2, a3)
    return np.sum(((y - y_pred) / errors)**2)

# Valores iniciales (estimados de los problemas anteriores)
initial_guess = [80, 75, (50**2)/4]  # [f_r, E_r, Γ²/4]

# Minimización usando el método de Nelder-Mead (robusto para problemas no lineales)
result = minimize(chi_square, initial_guess, args=(E, f, sigma), 
                  method='Nelder-Mead', options={'maxiter': 1000})

# Extraer parámetros optimizados
a1_opt, a2_opt, a3_opt = result.x
Gamma_opt = 2 * np.sqrt(a3_opt)

print("=== AJUSTE BREIT-WIGNER ===")
print(f"Parámetros optimizados:")
print(f"f_r = {a1_opt:.2f} MeV")
print(f"E_r = {a2_opt:.2f} MeV") 
print(f"Γ = {Gamma_opt:.2f} MeV")
print(f"Γ²/4 = {a3_opt:.2f} MeV²")
print(f"Chi-cuadrado mínimo: {result.fun:.2f}")
print(f"Número de iteraciones: {result.nit}")

# Graficar resultados
E_plot = np.linspace(0, 200, 500)
f_fit = breit_wigner(E_plot, a1_opt, a2_opt, a3_opt)

plt.figure(figsize=(12, 8))
plt.errorbar(E, f, yerr=sigma, fmt='o', label='Datos experimentales', 
             capsize=5, markersize=8, alpha=0.7)
plt.plot(E_plot, f_fit, 'r-', linewidth=2.5, 
         label=f'Breit-Wigner: $E_r$ = {a2_opt:.1f} MeV, $\Gamma$ = {Gamma_opt:.1f} MeV')
plt.xlabel('Energía (MeV)', fontsize=14)
plt.ylabel('f(E) (MeV)', fontsize=14)
plt.title('Ajuste Breit-Wigner para la Sección Eficaz de Resonancia', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Comparación con métodos anteriores
print("\n=== COMPARACIÓN CON MÉTODOS ANTERIORES ===")
print("Método           | E_r (MeV) | Γ (MeV) | Chi-cuadrado")
print("----------------------------------------------------")
print(f"Lagrange         |   75.0    |  50.0   |     -")
print(f"Splines          |   77.3    |  53.5   |     -") 
print(f"Breit-Wigner     |   {a2_opt:.1f}    |  {Gamma_opt:.1f}   |   {result.fun:.2f}")
print(f"Valor teórico    |   78.0    |  55.0   |     -")



#=============================================================================


# Datos
E = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
f = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])
sigma = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])

def breit_wigner(x, a1, a2, a3):
    return a1 / ((x - a2)**2 + a3)

def chi_square(params):
    a1, a2, a3 = params
    y_pred = breit_wigner(E, a1, a2, a3)
    return np.sum(((f - y_pred) / sigma)**2)

# Gradiente numérico automático
def gradient(params):
    return approx_fprime(params, chi_square, 1e-8)

# Guess inicial
initial_guess = [85.0, 75.0, (55.0**2)/4]

print("=== NEWTON-RAPHSON (Newton-CG) ===")
result = minimize(chi_square, initial_guess, method='Newton-CG',
                 jac=gradient, options={'disp': True})

a1_opt, a2_opt, a3_opt = result.x
Gamma_opt = 2 * np.sqrt(a3_opt)

print(f"f_r = {a1_opt:.2f} MeV")
print(f"E_r = {a2_opt:.2f} MeV")
print(f"Γ = {Gamma_opt:.2f} MeV")
print(f"χ² = {result.fun:.2f}")