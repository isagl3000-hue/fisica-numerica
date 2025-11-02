#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 11:40:48 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Definición de la función del paquete de onda gaussiano
def ww(x, t, k0, a0, vp, vg):
    """
    Calcula el paquete de onda gaussiano dispersivo
    
    Parámetros:
    x: array de posiciones
    t: tiempo
    k0: número de onda inicial
    a0: ancho inicial del paquete
    vp: velocidad de fase
    vg: velocidad de grupo
    
    Retorna:
    Parte real del paquete de onda normalizado
    """
    # tc = α + iβt, donde α = a0² y β = vg/(2k0)
    tc = a0*a0 + 1j*(0.5*vg/k0)*t
    
    # Paquete de onda: exp(i*k0*(x-vp*t)) * exp(-(x-vg*t)²/(4*tc))
    u = np.exp(1.0j*k0*(x - vp*t) - 0.25*(x - vg*t)**2 / tc)
    
    # Retornar la parte real del paquete normalizado
    return np.real(u / np.sqrt(tc))

# PARÁMETROS FÍSICOS DEL SISTEMA
wavelength = 1.0      # Longitud de onda inicial
a0 = 1.0              # Ancho inicial del paquete gaussiano
k0 = 2*np.pi/wavelength  # Número de onda (k = 2π/λ)
vp, vg = 5.0, 10.0    # Velocidades de fase y grupo

# PARÁMETROS TEMPORALES DE LA SIMULACIÓN
period = wavelength/vp           # Período de la onda
runtime = 40*period              # Tiempo total de simulación (40 períodos)
rundistance = 0.6*vg*runtime     # Distancia total a graficar
dt = period/6.0                  # Intervalo de tiempo entre frames
tsteps = int(runtime/dt)         # Número total de frames

# Información sobre la animación
print('Frame time interval = {0:0.3g} ms'.format(1000*dt))
print('Frame rate = {0:0.3g} frames/s'.format(1.0/dt))

# CONFIGURACIÓN DE LA FIGURA
fig, ax = plt.subplots(figsize=(12, 3))  # Crear figura con dimensiones 12x3 pulgadas

# Ajustar márgenes: bottom=0.2 deja espacio para la etiqueta del eje x
fig.subplots_adjust(bottom=0.2)

# Crear malla espacial para calcular la onda
# Desde -5*a0 hasta rundistance, con paso de wavelength/20
x = np.arange(-5*a0, rundistance, wavelength/20.0)

# Crear línea inicial vacía para la animación
# np.ma.array(x, mask=True) crea un array con todos los elementos enmascarados (invisibles)
line, = ax.plot(x, np.ma.array(x, mask=True), color='r')

# CONFIGURACIÓN DE LOS EJES
ax.set_xlabel(r'$x$')                    # Etiqueta del eje x en formato LaTeX
ax.set_ylabel(r'$y(x,t)$')               # Etiqueta del eje y en formato LaTeX
ax.set_xlim(-5*a0, rundistance)          # Límites del eje x
ax.set_ylim(-1.05, 1.05)                 # Límites del eje y (-1.05 a 1.05)

# FUNCIÓN DE ANIMACIÓN - se ejecuta para cada frame
def animate(i):
    """
    Función que actualiza la animación en cada frame
    
    Parámetro:
    i: número del frame actual
    """
    # Calcular el tiempo actual
    t = float(i) * dt
    
    # Actualizar los datos y (amplitud de la onda) usando la función ww
    line.set_ydata(ww(x, t, k0, a0, vp, vg))
    
    # Retornar los artistas que deben ser actualizados
    return line,

# CREAR LA ANIMACIÓN
ani = anim.FuncAnimation(
    fig,                    # Figura donde se dibuja la animación
    func=animate,           # Función que actualiza cada frame
    frames=range(tsteps),   # Secuencia de frames (0, 1, 2, ..., tsteps-1)
    interval=1000*dt,       # Intervalo entre frames en milisegundos
    blit=True               # Optimización: solo redibuja lo que cambia
)

# Mostrar la animación
fig.show()
