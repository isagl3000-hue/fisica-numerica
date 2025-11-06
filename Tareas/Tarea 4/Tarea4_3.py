#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 19:56:13 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# =============================================================================
# PARÁMETROS FÍSICOS
# =============================================================================
L = 1.0           # Longitud de la cuerda (m)
T = 100.0         # Tensión (N)
rho = 1.0         # Densidad lineal (kg/m)
c = np.sqrt(T/rho) # Velocidad de la onda (m/s)

print("=== PROBLEMA 3: VIBRACIÓN DE CUERDA ===")
print(f"Velocidad de la onda: c = {c:.2f} m/s")

# =============================================================================
# INCISO (d)-(f): MÉTODO DE DIFERENCIAS FINITAS
# =============================================================================
def resolver_cuerda_simple(dx, dt, tiempo_total, c, L):
    """
    Resuelve la ecuación de onda por diferencias finitas
    """
    Nx = int(L/dx) + 1
    Nt = int(tiempo_total/dt) + 1
    
    # Matriz solución
    y = np.zeros((Nx, Nt))
    x = np.linspace(0, L, Nx)
    
    # Condición de Courant
    r = (c * dt / dx)**2
    print(f"Condición de Courant: c*Δt/Δx = {np.sqrt(r):.3f}")
    
    if r > 1:
        print("¡ADVERTENCIA: La simulación puede ser inestable!")
    
    # Condición inicial: pulso triangular
    for i in range(Nx):
        if x[i] < L/2:
            y[i, 0] = 2*x[i]/L
        else:
            y[i, 0] = 2*(1 - x[i]/L)
    
    # Primera derivada temporal (velocidad inicial cero)
    y[:, 1] = y[:, 0]
    
    # Iteración temporal
    for j in range(1, Nt-1):
        for i in range(1, Nx-1):
            y[i, j+1] = 2*y[i, j] - y[i, j-1] + r*(y[i+1, j] + y[i-1, j] - 2*y[i, j])
    
    return x, np.linspace(0, tiempo_total, Nt), y

# =============================================================================
# INCISO (i): ANIMACIÓN OPTIMIZADA (basada en el codigo de la actividad 7)
# =============================================================================
def crear_animacion_cuerda(x, t, y, titulo):
    """
    Crea animación de la cuerda vibrante usando el estilo de tu código
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.subplots_adjust(bottom=0.2)
    
    # Configurar ejes
    ax.set_xlabel('Posición x (m)')
    ax.set_ylabel('Desplazamiento y (m)')
    ax.set_xlim(0, L)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Crear línea inicial
    line, = ax.plot(x, y[:, 0], color='b', linewidth=2)
    
    # Información de tiempo
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate_frame(i):
        """Actualiza cada frame de la animación"""
        line.set_ydata(y[:, i])
        time_text.set_text(f't = {t[i]:.3f} s')
        return line, time_text
    
    # Calcular parámetros de animación
    dt_anim = t[1] - t[0]
    interval = 50  # ms entre frames
    
    # Crear animación
    animation = anim.FuncAnimation(
        fig, 
        func=animate_frame,
        frames=len(t),
        interval=interval,
        blit=True
    )
    
    plt.show()
    return animation

# =============================================================================
# SIMULACIÓN PRINCIPAL
# =============================================================================

# CASO 1: CONDICIÓN DE COURANT SATISFECHA
print("\n--- CASO 1: Condición de Courant SATISFECHA ---")
dx1 = 0.01
dt1 = 0.0008  # c*dt/dx = 10*0.0008/0.01 = 0.8 ≤ 1
tiempo_total1 = 0.5

x1, t1, y1 = resolver_cuerda_simple(dx1, dt1, tiempo_total1, c, L)
anim1 = crear_animacion_cuerda(x1, t1, y1, "Cuerda - Condición Courant Satisfecha")

# CASO 2: CONDICIÓN DE COURANT VIOLADA
print("\n--- CASO 2: Condición de Courant VIOLADA ---")
dx2 = 0.01
dt2 = 0.002   # c*dt/dx = 10*0.002/0.01 = 2.0 > 1
tiempo_total2 = 0.1  # Tiempo más corto para evitar divergencia

x2, t2, y2 = resolver_cuerda_simple(dx2, dt2, tiempo_total2, c, L)
anim2 = crear_animacion_cuerda(x2, t2, y2, "Cuerda - Condición Courant Violada")

# =============================================================================
# INCISO (j): ANÁLISIS DE ESTABILIDAD
# =============================================================================
print("\n--- ANÁLISIS DE ESTABILIDAD ---")

# Probar diferentes condiciones de Courant
condiciones = [
    (0.02, 0.001, 0.5, "Muy estable"),
    (0.01, 0.0005, 0.5, "Estable"),
    (0.01, 0.001, 1.0, "Límite"),
    (0.01, 0.002, 2.0, "Inestable"),
    (0.005, 0.001, 2.0, "Muy inestable")
]

print("Resultados de estabilidad:")
for dx, dt, courant, desc in condiciones:
    # Simulación rápida para verificar estabilidad
    try:
        Nx_test = int(L/dx) + 1
        Nt_test = min(100, int(0.1/dt) + 1)
        
        y_test = np.zeros((Nx_test, Nt_test))
        x_test = np.linspace(0, L, Nx_test)
        
        # Condición inicial
        for i in range(Nx_test):
            if x_test[i] < L/2:
                y_test[i, 0] = 2*x_test[i]/L
            else:
                y_test[i, 0] = 2*(1 - x_test[i]/L)
        
        y_test[:, 1] = y_test[:, 0]
        r_test = (c * dt / dx)**2
        
        # Pocas iteraciones
        for j in range(1, min(50, Nt_test-1)):
            for i in range(1, Nx_test-1):
                y_test[i, j+1] = 2*y_test[i, j] - y_test[i, j-1] + r_test*(y_test[i+1, j] + y_test[i-1, j] - 2*y_test[i, j])
        
        max_val = np.max(np.abs(y_test))
        estable = max_val < 10
        
        estado = "ESTABLE" if estable else "INESTABLE"
        print(f"dx={dx:.3f}, dt={dt:.4f}, c*Δt/Δx={courant:.1f} - {estado}")
        
    except Exception as e:
        print(f"dx={dx:.3f}, dt={dt:.4f} - ERROR: {e}")

# =============================================================================
# GRÁFICO COMPARATIVO
# =============================================================================
plt.figure(figsize=(12, 8))

# Evolución temporal en punto central
punto_central1 = len(x1)//2
punto_central2 = len(x2)//2

plt.subplot(2, 2, 1)
plt.plot(t1, y1[punto_central1, :], 'b-', linewidth=2)
plt.title('Caso Estable - Evolución en x = L/2')
plt.xlabel('Tiempo (s)')
plt.ylabel('Desplazamiento (m)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(t2, y2[punto_central2, :], 'r-', linewidth=2)
plt.title('Caso Inestable - Evolución en x = L/2')
plt.xlabel('Tiempo (s)')
plt.ylabel('Desplazamiento (m)')
plt.grid(True, alpha=0.3)

# Instantáneas
plt.subplot(2, 2, 3)
for i in range(0, len(t1), len(t1)//5):
    plt.plot(x1, y1[:, i], alpha=0.7, label=f't={t1[i]:.2f}s')
plt.title('Caso Estable - Instantáneas')
plt.xlabel('Posición (m)')
plt.ylabel('Desplazamiento (m)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
for i in range(0, min(len(t2), 5)):
    plt.plot(x2, y2[:, i], alpha=0.7, label=f't={t2[i]:.2f}s')
plt.title('Caso Inestable - Instantáneas')
plt.xlabel('Posición (m)')
plt.ylabel('Desplazamiento (m)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# VERIFICACIONES TEÓRICAS
# =============================================================================
print("\n--- VERIFICACIONES TEÓRICAS ---")
print("(b) Condiciones para ecuación de onda estándar:")
print("   - Tensión constante: dT/dx = 0")
print("   - Densidad constante: ρ(x) = constante")
print("   - Pequeñas oscilaciones: ∂y/∂x << 1")

print("\n(c) Condiciones para solución única:")
print("   - Condiciones de frontera: y(0,t) = 0, y(L,t) = 0")
print("   - Condiciones iniciales: y(x,0) = f(x), ∂y/∂t(x,0) = g(x)")

print(f"\nFrecuencia fundamental teórica: f₁ = c/(2L) = {c/(2*L):.3f} Hz")

print("\n=== SIMULACIÓN COMPLETADA ===")