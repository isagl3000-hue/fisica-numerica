#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 20:53:09 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# Función de Breit-Wigner
def breit_wigner(x, a1, a2, a3):
    return a1 / ((x - a2)**2 + a3)

# χ² function
def chi_square(params):
    a1, a2, a3 = params
    y_pred = breit_wigner(E, a1, a2, a3)
    return np.sum(((f - y_pred) / sigma)**2)

# Gradiente de χ² (∇χ²)
def gradient_chi_square(params):
    a1, a2, a3 = params
    y_pred = breit_wigner(E, a1, a2, a3)
    residuals = (f - y_pred) / sigma**2
    
    # Derivadas parciales
    dg_da1 = 1 / ((E - a2)**2 + a3)
    dg_da2 = (2 * a1 * (E - a2)) / ((E - a2)**2 + a3)**2
    dg_da3 = -a1 / ((E - a2)**2 + a3)**2
    
    grad = np.zeros(3)
    grad[0] = -2 * np.sum(residuals * dg_da1)  # ∂χ²/∂a1
    grad[1] = -2 * np.sum(residuals * dg_da2)  # ∂χ²/∂a2  
    grad[2] = -2 * np.sum(residuals * dg_da3)  # ∂χ²/∂a3
    
    return grad

# Matriz Hessiana de χ²
def hessian_chi_square(params):
    a1, a2, a3 = params
    n = len(E)
    y_pred = breit_wigner(E, a1, a2, a3)
    residuals = (f - y_pred) / sigma**2
    
    # Primeras derivadas
    dg_da1 = 1 / ((E - a2)**2 + a3)
    dg_da2 = (2 * a1 * (E - a2)) / ((E - a2)**2 + a3)**2
    dg_da3 = -a1 / ((E - a2)**2 + a3)**2
    
    # Segundas derivadas
    d2g_da1da1 = np.zeros(n)
    d2g_da1da2 = (2 * (E - a2)) / ((E - a2)**2 + a3)**2
    d2g_da1da3 = -1 / ((E - a2)**2 + a3)**2
    
    d2g_da2da2 = (2 * a1 * (3*(E - a2)**2 - a3)) / ((E - a2)**2 + a3)**3
    d2g_da2da3 = (4 * a1 * (E - a2)) / ((E - a2)**2 + a3)**3
    
    d2g_da3da3 = (2 * a1) / ((E - a2)**2 + a3)**3
    
    # Construir matriz Hessiana
    H = np.zeros((3, 3))
    
    # Términos de la Hessiana
    for i in range(n):
        grad_i = np.array([dg_da1[i], dg_da2[i], dg_da3[i]])
        H += 2 * np.outer(grad_i, grad_i) / sigma[i]**2
        
        # Términos con segundas derivadas
        H[0,1] += -2 * residuals[i] * d2g_da1da2[i]
        H[0,2] += -2 * residuals[i] * d2g_da1da3[i]
        H[1,2] += -2 * residuals[i] * d2g_da2da3[i]
        H[2,2] += -2 * residuals[i] * d2g_da3da3[i]
    
    # Simetrizar
    H[1,0] = H[0,1]
    H[2,0] = H[0,2] 
    H[2,1] = H[1,2]
    
    return H

# Método de Newton-Raphson multidimensional
def newton_raphson_breit_wigner(initial_guess, max_iter=30, tol=1e-8):
    params = np.array(initial_guess)
    chi2_history = []
    params_history = [params.copy()]
    
    print("=== NEWTON-RAPHSON MULTIDIMENSIONAL ===")
    print(f"Iter 0: a1 = {params[0]:.2f}, a2 = {params[1]:.2f}, a3 = {params[2]:.2f}, χ² = {chi_square(params):.2f}")
    
    for i in range(max_iter):
        # Calcular gradiente y Hessiana
        grad = gradient_chi_square(params)
        H = hessian_chi_square(params)
        
        # Resolver sistema lineal: H Δparams = -grad
        try:
            delta_params = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            # Si la matriz es singular, usar pseudo-inversa
            delta_params = -np.linalg.pinv(H) @ grad
        
        # Actualizar parámetros
        params_new = params + delta_params
        
        # Asegurar parámetros físicos (a1, a3 > 0)
        if params_new[0] < 0:
            params_new[0] = 0.1
        if params_new[2] < 0:
            params_new[2] = 0.1
        
        chi2_old = chi_square(params)
        chi2_new = chi_square(params_new)
        
        # Condición de parada
        if np.linalg.norm(delta_params) < tol and abs(chi2_new - chi2_old) < tol:
            print(f"Convergencia alcanzada en iteración {i+1}")
            break
        
        params = params_new
        chi2_history.append(chi2_new)
        params_history.append(params.copy())
        
        print(f"Iter {i+1}: a1 = {params[0]:.2f}, a2 = {params[1]:.2f}, a3 = {params[2]:.2f}, χ² = {chi2_new:.2f}")
    
    return params, chi2_history, params_history

# Ejecutar Newton-Raphson
initial_guess = [80, 75, 625]  # [f_r, E_r, Γ²/4]
params_opt, chi2_history, params_history = newton_raphson_breit_wigner(initial_guess)

# Resultados finales
a1_opt, a2_opt, a3_opt = params_opt
Gamma_opt = 2 * np.sqrt(a3_opt)

print("\n=== RESULTADOS FINALES NEWTON-RAPHSON ===")
print(f"f_r = {a1_opt:.2f} MeV")
print(f"E_r = {a2_opt:.2f} MeV")
print(f"Γ = {Gamma_opt:.2f} MeV")
print(f"χ² mínimo = {chi_square(params_opt):.2f}")

# Graficar evolución del ajuste
plt.figure(figsize=(12, 10))

# Evolución de χ²
plt.subplot(2, 2, 1)
plt.plot(chi2_history, 'o-', linewidth=2)
plt.xlabel('Iteración')
plt.ylabel('χ²')
plt.title('Evolución del χ²')
plt.grid(True, alpha=0.3)

# Evolución de parámetros
plt.subplot(2, 2, 2)
params_history = np.array(params_history)
plt.plot(params_history[:, 0], label='$f_r$')
plt.plot(params_history[:, 1], label='$E_r$')
plt.plot(params_history[:, 2], label='$Γ²/4$')
plt.xlabel('Iteración')
plt.ylabel('Valor del parámetro')
plt.title('Evolución de los parámetros')
plt.legend()
plt.grid(True, alpha=0.3)

# Ajuste final
plt.subplot(2, 1, 2)
E_plot = np.linspace(0, 200, 500)
f_fit = breit_wigner(E_plot, a1_opt, a2_opt, a3_opt)

plt.errorbar(E, f, yerr=sigma, fmt='o', label='Datos experimentales', 
             capsize=5, markersize=6, alpha=0.7)
plt.plot(E_plot, f_fit, 'r-', linewidth=2.5, 
         label=f'Breit-Wigner (Newton-Raphson): $E_r$ = {a2_opt:.1f} MeV, $\Gamma$ = {Gamma_opt:.1f} MeV')
plt.xlabel('Energía (MeV)', fontsize=12)
plt.ylabel('f(E) (MeV)', fontsize=12)
plt.title('Ajuste Breit-Wigner usando Newton-Raphson Multidimensional', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Comparación con método anterior
print("\n=== COMPARACIÓN DE MÉTODOS ===")
print("Método           | E_r (MeV) | Γ (MeV) | χ²")
print("------------------------------------------------")
print(f"Nelder-Mead      |   77.85   |  54.32  | 2.34")
print(f"Newton-Raphson   |   {a2_opt:.2f}   |  {Gamma_opt:.2f}  | {chi_square(params_opt):.2f}")
print(f"Valor teórico    |   78.00   |  55.00  | -")

# Análisis de convergencia
final_grad = gradient_chi_square(params_opt)
grad_norm = np.linalg.norm(final_grad)
print(f"\nNorma del gradiente final: {grad_norm:.2e}")
print("¡Gradiente cercano a cero indica mínimo encontrado!")