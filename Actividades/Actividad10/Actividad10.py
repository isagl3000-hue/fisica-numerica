#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 11:08:25 2025

@author: isaias-gl
"""

import numpy as np
import pandas as pd

def simulacion_embarazo(n_simulaciones):
    """
    Simula el número de años hasta el primer embarazo
    usando el método de Monte Carlo
    """
    años_hasta_embarazo = []
    
    for _ in range(n_simulaciones):
        años = 0
        embarazada = False
        
        while not embarazada:
            años += 1
            # Generar un número aleatorio entre 0 y 1
            # Si es menor que 0.3, queda embarazada (30% de probabilidad)
            if np.random.random() <= 0.3:
                embarazada = True
        
        años_hasta_embarazo.append(años)
    
    return np.array(años_hasta_embarazo)

valores_n = [1000, 10000, 100000]

# Almacenar resultados
resultados = []
# Versión alternativa más eficiente usando distribución geométrica
print("\n" + "=" * 50)
print("VERSIÓN ALTERNATIVA MÁS EFICIENTE")
print("=" * 50)
print("Simulaciones de Monte Carlo - Método del Ritmo")
print("=" * 50)
print(f"Probabilidad de embarazo por año: 30%")
print(f"Valor teórico esperado: {1/0.3:.4f} años")
print()

for n in valores_n:
    print(f"Simulando n = {n} experimentos...")
    
    # Ejecutar simulación
    resultados_simulacion = simulacion_embarazo(n)
    
    # Calcular estadísticas
    media = np.mean(resultados_simulacion)
    desviacion = np.std(resultados_simulacion)
    
    # Almacenar para la tabla
    resultados.append({
        'n': n,
        'media': media,
        'desviacion_estandar': desviacion
    })
    
    print(f"  Media: {media:.4f} años")
    print(f"  Desviación estándar: {desviacion:.4f}")
    print()

# Crear tabla de resultados
tabla_resultados = pd.DataFrame(resultados)
print("TABLA DE RESULTADOS")
print("=" * 40)
print(tabla_resultados.to_string(index=False, float_format='%.4f'))






