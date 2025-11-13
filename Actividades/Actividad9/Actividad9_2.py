#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:52:34 2025

@author: isaias-gl
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuración
np.random.seed(42)  # Para reproducibilidad
random.seed(42)

def simular_lanzamientos_moneda(n):
    """
    Simula n lanzamientos de una moneda justa
    Returns: Array con 1 para cara y 0 para cruz
    """
    # Usamos numpy para mayor eficiencia con grandes cantidades de datos
    return np.random.choice([0, 1], size=n)

def analizar_simulacion(n):

    lanzamientos = simular_lanzamientos_moneda(n)
    
    # Cálculo de estadísticas
    media = np.mean(lanzamientos)
    desviacion_estandar = np.std(lanzamientos)
    varianza = np.var(lanzamientos)
    
    # Conteo de caras y cruces
    caras = np.sum(lanzamientos)
    cruces = n - caras
    
    return {
        'n_lanzamientos': n,
        'media': media,
        'desviacion_estandar': desviacion_estandar,
        'varianza': varianza,
        'caras': caras,
        'cruces': cruces,
        'proporcion_caras': caras / n,
        'proporcion_cruces': cruces / n
    }


# 1. Realizar simulaciones para diferentes valores de n
valores_n = [100, 1000, 10000]
resultados = []

print("Simulando lanzamientos de moneda...")
print("=" * 50)

for n in valores_n:
    resultado = analizar_simulacion(n)
    resultados.append(resultado)
    
    print(f"\nResultados para n = {n}:")
    print(f"  Media: {resultado['media']:.4f}")
    print(f"  Desviación estándar: {resultado['desviacion_estandar']:.4f}")
    print(f"  Varianza: {resultado['varianza']:.4f}")
    print(f"  Caras: {resultado['caras']} ({resultado['proporcion_caras']*100:.2f}%)")
    print(f"  Cruces: {resultado['cruces']} ({resultado['proporcion_cruces']*100:.2f}%)")
    






