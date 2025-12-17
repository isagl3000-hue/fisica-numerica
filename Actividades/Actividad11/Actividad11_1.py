#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 11:12:42 2025

@author: isaias-gl
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def monte_carlo_integral(func, a, b, n_points=10000):
    """
    Calcula la integral de una función usando el método de Monte Carlo
    """
    # Generar puntos aleatorios uniformemente distribuidos
    x_random = np.random.uniform(a, b, n_points)
    
    # Evaluar la función en los puntos aleatorios
    f_values = func(x_random)
    
    # Calcular la integral estimada
    integral_estimate = (b - a) * np.mean(f_values)
    
    return integral_estimate

def f1(x):
    return (1 - x**2)**3

def f2(x):
    """Función para la integral (b): e^(x + x²)"""
    return np.exp(x + x**2)

def analitica_f1():
    """Solución analítica de la integral (a)"""
    # ∫(1 - x²)³ dx = ∫(1 - 3x² + 3x⁴ - x⁶) dx
    # = x - x³ + (3/5)x⁵ - (1/7)x⁷
    return 1 - 1 + (3/5) - (1/7)

def analitica_f2():
    """Solución analítica de la integral (b) requiere método numérico"""
    result, error = integrate.quad(f2, -2, 2)
    return result


def estudio_convergencia(func, a, b, analitica):
    """
    Estudia la convergencia del método de Monte Carlo
    """
    # Valores de N para estudiar la convergencia
    N_values = [100, 500, 1000, 5000, 10000, 50000, 100000]
    
    estimaciones = []
    errores = []
    
    
    print("="*50)
    
    for N in N_values:
        estimacion = monte_carlo_integral(func, a, b, N)
        estimaciones.append(estimacion)
        error = abs(estimacion - analitica)
        errores.append(error)
        
        print(f"N = {N:6d}: Estimado = {estimacion:.6f}, Error = {error:.6f}")
    
    # Gráfica de convergencia
    plt.figure(figsize=(10, 6))
    
    
    plt.semilogx(N_values, estimaciones, 's-', linewidth=2, markersize=8, 
                label='Monte Carlo')
    plt.axhline(y=analitica, color='red', linestyle='--', 
               label=f'Analítico = {analitica:.6f}')
    plt.xlabel('Número de puntos (N)')
    plt.ylabel('Valor estimado')
    plt.title(f'Estimación vs N')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return estimaciones, errores

# Ejecutar el estudio para ambas integrales
if __name__ == "__main__":
    print("INTEGRACIÓN POR MÉTODO DE MONTE CARLO")
    print("="*60)
    
    # Integral (a): ∫₀¹ (1 - x²)³ dx
    analitica_a = analitica_f1()
    print(f"\nIntegral (a):")
    print(f"Valor analítico: {analitica_a:.10f}")
    
    estudio_convergencia(f1, 0, 1, analitica_a)
    
    # Integral (b): ∫₋₂² e^(x + x²) dx
    analitica_b = analitica_f2()
    print(f"\nIntegral (b):")
    print(f"Valor analítico (scipy): {analitica_b:.10f}")
    
    estudio_convergencia(f2, -2, 2, analitica_b)
