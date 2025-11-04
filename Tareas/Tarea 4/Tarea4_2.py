#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:52:25 2025

@author: isaias-gl
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# =============================================================================
# INCISO (a): Ecuaciones de movimiento
# =============================================================================

def ecuaciones_lineales(t, y, k, m):
    """
    Ecuaciones de movimiento para el sistema lineal
    y = [x1, x2, v1, v2]
    """
    x1, x2, v1, v2 = y
    
    # Fuerzas sobre masa 1: -k*x1 (resorte izquierdo) + k*(x2 - x1) (resorte central)
    a1 = (-k*x1 + k*(x2 - x1)) / m
    
    # Fuerzas sobre masa 2: -k*x2 (resorte derecho) + k*(x1 - x2) (resorte central)  
    a2 = (-k*x2 + k*(x1 - x2)) / m
    
    return [v1, v2, a1, a2]

def ecuaciones_no_lineales(t, y, k, m):
    """
    Ecuaciones de movimiento para el sistema no lineal
    F = -k(x + 0.1x³)
    """
    x1, x2, v1, v2 = y
    
    # Fuerzas no lineales
    F1_left = -k * (x1 + 0.1 * x1**3)
    F1_center = -k * ((x2 - x1) + 0.1 * (x2 - x1)**3)
    F2_right = -k * (x2 + 0.1 * x2**3)
    F2_center = -k * ((x1 - x2) + 0.1 * (x1 - x2)**3)
    
    a1 = (F1_left + F1_center) / m
    a2 = (F2_right + F2_center) / m
    
    return [v1, v2, a1, a2]

# =============================================================================
# INCISO (b): Frecuencias de modos normales
# =============================================================================

def calcular_modos_normales(k, m):
    """
    Calcula las frecuencias y modos normales del sistema
    """
    # Matriz de ecuaciones
    A = np.array([[2*k, -k], 
                  [-k, 2*k]]) / m
    
    # Valores propios y vectores propios
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Frecuencias angulares (sqrt de valores propios)
    frecuencias = np.sqrt(eigenvalues)
    
    modos = {
        'frecuencias': frecuencias,
        'modos': eigenvectors.T,  # Transponer para tener modos en filas
        'modo_simetrico': eigenvectors[:, 0],
        'modo_antisimetrico': eigenvectors[:, 1]
    }
    
    return modos

# =============================================================================
# INCISO (c): Simulaciones de diferentes condiciones iniciales
# =============================================================================

def simular_sistema(condiciones_iniciales, k=1.0, m=1.0, no_lineal=False, t_max=100):
    """
    Simula el sistema para condiciones iniciales dadas
    """
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 1000)
    
    if no_lineal:
        sol = solve_ivp(ecuaciones_no_lineales, t_span, condiciones_iniciales, 
                       args=(k, m), t_eval=t_eval, method='RK45')
    else:
        sol = solve_ivp(ecuaciones_lineales, t_span, condiciones_iniciales, 
                       args=(k, m), t_eval=t_eval, method='RK45')
    
    return sol

def graficar_resultados(sol, titulo):
    """
    Grafica las posiciones de las masas
    """
    plt.figure(figsize=(12, 4))
    
    # Posiciones vs tiempo
    plt.subplot(1, 2, 1)
    plt.plot(sol.t, sol.y[0], 'b-', label='Masa 1', linewidth=2)
    plt.plot(sol.t, sol.y[1], 'r-', label='Masa 2', linewidth=2)
    plt.xlabel('Tiempo')
    plt.ylabel('Posición')
    plt.title(f'{titulo} - Posiciones vs Tiempo')
    plt.legend()
    plt.grid(True)
    
    # Espacio de fases
    plt.subplot(1, 2, 2)
    plt.plot(sol.y[0], sol.y[2], 'b-', label='Masa 1', alpha=0.7)
    plt.plot(sol.y[1], sol.y[3], 'r-', label='Masa 2', alpha=0.7)
    plt.xlabel('Posición')
    plt.ylabel('Velocidad')
    plt.title(f'{titulo} - Espacio de Fases')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Parámetros del sistema
k = 1.0  # constante del resorte
m = 1.0  # masa

print("=== PROBLEMA 2: SISTEMA DE MASAS Y RESORTES ===")

# =============================================================================
# INCISO (b): Cálculo de modos normales
# =============================================================================
print("\n--- INCISO (b): Modos Normales ---")
modos = calcular_modos_normales(k, m)

print(f"Frecuencias angulares: {modos['frecuencias']}")
print(f"Modo simétrico (baja frecuencia): {modos['modo_simetrico']}")
print(f"Modo antisimétrico (alta frecuencia): {modos['modo_antisimetrico']}")

# =============================================================================
# INCISO (c): Simulaciones con diferentes condiciones iniciales
# =============================================================================
print("\n--- INCISO (c): Simulaciones ---")

# Condición i: Ambas masas desplazadas igual hacia la derecha
print("Caso i: Ambas masas desplazadas igual hacia la derecha")
cond_i = [0.5, 0.5, 0, 0]  # [x1, x2, v1, v2]
sol_i = simular_sistema(cond_i, k, m)
graficar_resultados(sol_i, "Caso i - Desplazamiento igual")

# Condición ii: Masas desplazadas en sentidos opuestos
print("Caso ii: Masas desplazadas en sentidos opuestos")
cond_ii = [0.5, -0.5, 0, 0]
sol_ii = simular_sistema(cond_ii, k, m)
graficar_resultados(sol_ii, "Caso ii - Desplazamiento opuesto")

# Condición iii: Una masa en equilibrio, otra desplazada
print("Caso iii: Una masa en equilibrio, otra desplazada")
cond_iii = [0.0, 0.5, 0, 0]
sol_iii = simular_sistema(cond_iii, k, m)
graficar_resultados(sol_iii, "Caso iii - Una masa desplazada")

# =============================================================================
# INCISO (d): Comparación lineal vs no lineal
# =============================================================================
print("\n--- INCISO (d): Lineal vs No Lineal ---")

# Simular caso iii con sistema no lineal
print("Comparando caso iii en sistemas lineal y no lineal...")
sol_iii_lineal = simular_sistema(cond_iii, k, m, no_lineal=False)
sol_iii_no_lineal = simular_sistema(cond_iii, k, m, no_lineal=True)

# Graficar comparación
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(sol_iii_lineal.t, sol_iii_lineal.y[1], 'b-', label='Lineal', linewidth=2)
plt.ylabel('Posición Masa 2')
plt.title('Comparación: Sistema Lineal vs No Lineal')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(sol_iii_no_lineal.t, sol_iii_no_lineal.y[1], 'r-', label='No lineal', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Posición Masa 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Análisis de diferencias
print("\nAnálisis de diferencias:")
frec_lineal = 1/(sol_iii_lineal.t[1] - sol_iii_lineal.t[0]) * 0.1  # Estimación frecuencia
print(f"Frecuencia aproximada sistema lineal: {frec_lineal:.3f} Hz")

