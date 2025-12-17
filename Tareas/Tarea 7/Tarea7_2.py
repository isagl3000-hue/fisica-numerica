#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 20:21:49 2025

@author: isaias-gl
"""

import random
import math
import scipy.stats as stats

def frecuencia_test(n=10000, k=10, seed=None):
    """
    Aplica la prueba de frecuencia (Chi-cuadrada) a random.random().
    
    Parámetros:
    n: número de muestras
    k: número de categorías (subintervalos)
    seed: semilla para reproducibilidad
    
    Retorna:
    V: estadístico Chi-cuadrado
    p_value: probabilidad de obtener un valor tan extremo
    frec_obs: frecuencias observadas
    frec_esp: frecuencias esperadas
    """
    if seed is not None:
        random.seed(seed)
    
    # Generar n números en [0,1)
    numeros = [random.random() for _ in range(n)]
    
    # Dividir en k categorías iguales
    categorias = [0] * k
    for x in numeros:
        idx = int(x * k)  # índice de la categoría
        if idx == k:      # por si x == 1.0 (muy raro)
            idx = k - 1
        categorias[idx] += 1
    
    # Frecuencias esperadas
    esperado = n / k
    
    # Calcular estadístico Chi-cuadrado
    V = sum((obs - esperado)**2 / esperado for obs in categorias)
    
    # Grados de libertad = k - 1
    p_value = 1 - stats.chi2.cdf(V, k - 1)
    
    return V, p_value, categorias, [esperado] * k

# Aplicar la prueba
if __name__ == "__main__":
    n = 10000
    k = 10
    seed = 12345  # semilla fija para reproducibilidad
    
    V, p, obs, esp = frecuencia_test(n, k, seed)
    
    # Mostrar resultados
    print("="*60)
    print("PRUEBA DE FRECUENCIA PARA RANDOM.RANDOM()")
    print("="*60)
    print(f"Número de muestras: n = {n}")
    print(f"Número de categorías: k = {k}")
    print(f"Grados de libertad: df = {k-1}")
    print("-"*60)
    print("Categoría   Observado   Esperado   Diferencia")
    for i in range(k):
        print(f"{i+1:>6}      {obs[i]:>8}    {esp[i]:>8.1f}    {obs[i]-esp[i]:>10.1f}")
    print("-"*60)
    print(f"Estadístico Chi-cuadrado: V = {V:.4f}")
    print(f"Valor p: p = {p:.6f}")
    
    # Decisión
    alpha = 0.05  # nivel de significancia
    if p < alpha:
        print("DECISIÓN: Rechazamos la hipótesis de uniformidad (no aleatorio).")
    else:
        print("DECISIÓN: No hay evidencia para rechazar la uniformidad (aleatorio).")
    
    # Valor crítico
    critico = stats.chi2.ppf(1 - alpha, k-1)
    print(f"Valor crítico (alpha={alpha}): {critico:.4f}")
    if V > critico:
        print("V > crítico → Rechazar hipótesis de uniformidad.")
    else:
        print("V ≤ crítico → No rechazar hipótesis de uniformidad.")
    print("="*60)