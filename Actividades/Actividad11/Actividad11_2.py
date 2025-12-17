#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 11:14:51 2025

@author: isaias-gl
"""


import numpy as np
import matplotlib.pyplot as plt
from pylab import *

lambda1 = 0.1    # Actividad
max_nucleos = 200.0
tiempo_max = 300
numero = nloop = max_nucleos  # Valor inicial

# Arrays para guardar los resultados
tiempos = []
nucleos_simulados = []
nucleos_analiticos = []

print("Tiempo | Simulados | Analítico")
print("-" * 30)

for tiempo in arange(0, tiempo_max + 1):  # Ciclo del tiempo
    # Ciclo del decaimiento para cada núcleo
    for nucleo in arange(1, numero + 1):
        decaimiento = np.random.rand()
        if decaimiento < lambda1:
            nloop -= 1
            # winsound.Beep(900,100)  # Eliminado por compatibilidad
    
    numero = nloop
    
    # Calcular valor analítico (fórmula exacta)
    N_analitico = max_nucleos * np.exp(-lambda1 * tiempo)
    
    # Guardar resultados
    tiempos.append(tiempo)
    nucleos_simulados.append(numero)
    nucleos_analiticos.append(N_analitico)
    
    # Mostrar algunos valores
    if tiempo % 50 == 0:
        print(f"{tiempo:6.0f} | {numero:9.0f} | {N_analitico:8.1f}")
    
    # Graficar punto de la simulación
    plt.plot(tiempo, numero, '*', color='r')
    
    # Graficar punto del modelo analítico
    plt.plot(tiempo, N_analitico, 's', color='b', markersize=3)

# Al final, hacer la gráfica completa
plt.xlabel('Tiempo')
plt.ylabel('Número de núcleos')
plt.title('Decaimiento Radioactivo: Simulación (rojo) vs Analítico (azul)')
plt.grid(True)

# Agregar leyenda
plt.plot([], [], '*', color='r', label='Simulación Monte Carlo')
plt.plot([], [], 's', color='b', label='Modelo Analítico: N(t) = N₀e^(-λt)')
plt.legend()

plt.show()

print("\nSimulación completada!")
print(f"Núcleos finales - Simulación: {numero}")
print(f"Núcleos finales - Analítico: {nucleos_analiticos[-1]:.1f}")