#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 21:27:31 2025

@author: isaias-gl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 20:48:22 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def solucion_analitica(x, t, L=1.0, T0=100.0, alpha=8.418e-5, n_terms=50):
    """
    Solución analítica de la ecuación de calor 1D
    """
    T_analitica = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        Bn = (2 * T0 / (n * np.pi)) * (1 - np.cos(n * np.pi))
        T_analitica += Bn * np.sin(n * np.pi * x / L) * np.exp(-alpha * (n * np.pi / L)**2 * t)
    return T_analitica

def resolver_calor_numerico(dx=0.02, dt=0.4, total_time=500.0):
    """
    Solución numérica por diferencias finitas
    """
    L = 1.0
    T0 = 100.0
    alpha = 8.418e-5
    
    Nx = int(L / dx) + 1
    Nt = int(total_time / dt) + 1
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, total_time, Nt)
    
    r = alpha * dt / (dx**2)
    
    # Solución numérica
    T_num = np.zeros((Nx, Nt))
    T_num[1:-1, 0] = T0
    T_num[0, :] = 0.0
    T_num[-1, :] = 0.0
    
    for j in range(0, Nt-1):
        for i in range(1, Nx-1):
            T_num[i, j+1] = T_num[i, j] + r * (T_num[i+1, j] - 2*T_num[i, j] + T_num[i-1, j])
    
    return x, t, T_num, r

# Parámetros de simulación
dx, dt = 0.02, 0.4
total_time = 300.0
x, t, T_num, r = resolver_calor_numerico(dx, dt, total_time)

# Crear mallas para solución analítica
X, T_mesh = np.meshgrid(x, t, indexing='ij')

# Calcular solución analítica
T_an = np.zeros_like(T_num)
for j in range(len(t)):
    T_an[:, j] = solucion_analitica(x, t[j])

# Calcular error
error = np.abs(T_num - T_an)

# Crear visualizaciones
fig = plt.figure(figsize=(20, 10))

# Gráfica 1: Solución numérica 3D (ESTABLE)
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X, T_mesh, T_num, cmap='hot', alpha=0.8)
ax1.contour(X, T_mesh, T_num, levels=10, offset=0, colors='black', linewidths=0.5)
ax1.set_xlabel('Posición (m)')
ax1.set_ylabel('Tiempo (s)')
ax1.set_zlabel('Temperatura (°C)')
ax1.set_title('SOLUCIÓN NUMÉRICA (ESTABLE)\nr = 0.42 < 0.5')
ax1.view_init(30, -45)

# Gráfica 2: Solución analítica 3D
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X, T_mesh, T_an, cmap='hot', alpha=0.8)
ax2.contour(X, T_mesh, T_an, levels=10, offset=0, colors='black', linewidths=0.5)
ax2.set_xlabel('Posición (m)')
ax2.set_ylabel('Tiempo (s)')
ax2.set_zlabel('Temperatura (°C)')
ax2.set_title('SOLUCIÓN ANALÍTICA 3D')
ax2.view_init(30, -45)

# Gráfica 3: Error y comparación 2D
ax3 = fig.add_subplot(2, 3, 3)

# Perfiles en tiempos específicos
times_to_plot = [0, 50, 100, 200]
colors = ['red', 'blue', 'green', 'orange']

for i, time_idx in enumerate(times_to_plot):
    if time_idx < len(t):
        ax3.plot(x, T_num[:, time_idx], color=colors[i], 
                linestyle='-', linewidth=2, label=f'Num t={t[time_idx]:.0f}s')
        ax3.plot(x, T_an[:, time_idx], color=colors[i], 
                linestyle='--', linewidth=1.5, alpha=0.7, label=f'An t={t[time_idx]:.0f}s')

ax3.set_xlabel('Posición (m)')
ax3.set_ylabel('Temperatura (°C)')
ax3.set_title('Comparación Numérica vs Analítica')
ax3.legend()
ax3.grid(True)

# NUEVAS GRÁFICAS: Solución estable e inestable en 2D

# Gráfica 4: Solución estable en diferentes tiempos
ax4 = fig.add_subplot(2, 3, 4)

for i, time_idx in enumerate(times_to_plot):
    if time_idx < len(t):
        ax4.plot(x, T_num[:, time_idx], color=colors[i], 
                linewidth=2, label=f't = {t[time_idx]:.0f}s')

ax4.set_xlabel('Posición (m)')
ax4.set_ylabel('Temperatura (°C)')
ax4.set_title('SOLUCIÓN ESTABLE\nEvolución Temporal')
ax4.legend()
ax4.grid(True)

# Gráfica 5: Solución inestable
ax5 = fig.add_subplot(2, 3, 5)

# Calcular solución inestable
dx_inestable, dt_inestable = 0.02, 4.0
x_inest, t_inest, T_inest, r_inest = resolver_calor_numerico(dx_inestable, dt_inestable, total_time)

# Seleccionar tiempos para graficar (menos puntos para evitar sobrecarga)
time_indices_inest = [0, min(10, len(t_inest)-1), min(20, len(t_inest)-1), min(30, len(t_inest)-1)]

for i, time_idx in enumerate(time_indices_inest):
    ax5.plot(x_inest, T_inest[:, time_idx], color=colors[i], 
             linewidth=2, label=f't = {t_inest[time_idx]:.0f}s')

ax5.set_xlabel('Posición (m)')
ax5.set_ylabel('Temperatura (°C)')
ax5.set_title(f'SOLUCIÓN INESTABLE\nr = {r_inest:.3f} > 0.5')
ax5.legend()
ax5.grid(True)


plt.tight_layout()
plt.show()
