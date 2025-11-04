import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def martillo(t, state, g, rho, A, CD, m):
    """
    Ecuaciones de movimiento del martillo con resistencia del aire
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    v = np.sqrt(vx**2 + vy**2)
    
    # Fuerza de arrastre (proporcional a v²)
    F_D = 0.5 * rho * A * CD * v**2
    
    # Componentes de la aceleración
    ax = - (F_D/m) * (vx/v) if v > 0 else 0
    ay = -g - (F_D/m) * (vy/v) if v > 0 else -g
    
    return [vx, vy, ax, ay]

# Parámetros físicos
g = 9.81  # m/s²
m = 7.26  # kg
R = 0.06  # m
A = np.pi * R**2  # m²
rho = 1.2  # kg/m³
phi = np.radians(45)  # ángulo inicial

# Coeficientes de arrastre
CD_casos = {
    'sin_friccion': 0.0,
    'laminar': 0.5,
    'inestable': 0.75
}

def encontrar_velocidad_record(CD, v0_guess=30, tol=0.1):
    """Encuentra la velocidad inicial que da la distancia del record"""
    v0 = v0_guess
    
    for _ in range(50):  # máximo 50 iteraciones
        # Condiciones iniciales
        state0 = [0, 2, v0 * np.cos(phi), v0 * np.sin(phi)]
        
        # Tiempo de integración
        t_span = (0, 15)
        t_eval = np.linspace(0, 15, 1000)
        
        # Resolver EDOs
        sol = solve_ivp(martillo, t_span, state0, args=(g, rho, A, CD, m), 
                       t_eval=t_eval, method='RK45')
        
        # Encontrar cuando y = 0 (martillo toca tierra)
        y_values = sol.y[1]
        x_values = sol.y[0]
        
        # Encontrar el índice donde y cruza por cero
        ground_index = np.where(y_values < 0)[0]
        if len(ground_index) > 0:
            ground_index = ground_index[0]
            distancia = x_values[ground_index]
        else:
            distancia = x_values[-1]
        
        # Ajustar velocidad
        error = distancia - 86.74
        if abs(error) < tol:
            break
            
        v0 = v0 - error * 0.1  # ajuste proporcional
    
    return v0, sol, distancia

# Encontrar velocidades para cada caso
resultados = {}
for nombre, CD in CD_casos.items():
    v0, sol, distancia = encontrar_velocidad_record(CD)
    resultados[nombre] = {
        'v0': v0,
        'solucion': sol,
        'distancia': distancia
    }
    print(f"{nombre:15}: v0 = {v0:.2f} m/s, distancia = {distancia:.2f} m")

# Graficar trayectorias
plt.figure(figsize=(12, 4))

# Trayectoria y vs x
plt.subplot(1, 2, 1)
for nombre, resultado in resultados.items():
    sol = resultado['solucion']
    x = sol.y[0]
    y = sol.y[1]
    plt.plot(x, y, label=nombre, linewidth=2)

plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.title('Trayectorias del martillo')
plt.legend()
plt.grid(True)

# Altura vs tiempo
plt.subplot(1, 2, 2)
for nombre, resultado in resultados.items():
    sol = resultado['solucion']
    t = sol.t
    y = sol.y[1]
    plt.plot(t, y, label=nombre, linewidth=2)

plt.xlabel('Tiempo (s)')
plt.ylabel('Altura (m)')
plt.title('Altura vs tiempo')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Calcular influencia de la fricción
v0_sin = resultados['sin_friccion']['v0']
v0_laminar = resultados['laminar']['v0']
v0_inestable = resultados['inestable']['v0']

print(f"\nInfluencia de la fricción:")
print(f"Diferencia velocidad laminar vs sin fricción: {v0_laminar - v0_sin:.2f} m/s")
print(f"Diferencia velocidad inestable vs sin fricción: {v0_inestable - v0_sin:.2f} m/s")#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 13:08:47 2025

@author: isaias-gl
"""

