#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:14:09 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt

# Datos de Hubble: distancia (r) en Mpc y velocidad (v) en km/s
r = np.array([0.032, 0.034, 0.214, 0.263, 0.275, 0.275, 0.450, 0.500, 
              0.500, 0.630, 0.800, 0.900, 0.900, 0.900, 0.900, 1.000, 
              1.100, 1.100, 1.400, 1.700, 2.000, 2.000, 2.000, 2.000])

v = np.array([170, 290, -130, -70, -185, -220, 200, 290, 270, 200, 
              300, -30, 650, 150, 500, 920, 450, 500, 500, 960, 500, 
              850, 800, 1090])

plt.plot(figsize=(10,6))
plt.scatter(r,v, color='blue', alpha=0.7, s=60, label='Datos Hubble')
plt.xlabel('DIstancia r (Mpc)', fontsize=12)
plt.ylabel('Velocidad v (Km/seg)', fontsize=12)
plt.title('Datos Hubble: Velocidad vs Distancia', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

sigma_v=0.1*np.abs(v) #Asumimos el 10% de error en cada medición
sigma_v=np.maximum(sigma_v,10) #Minimo 10 km/s de error

n=len(r)
sum_r=np.sum(r)
sum_v=np.sum(v)
sum_r2=np.sum(r**2)
sum_rv=np.sum(r*v)


H = (n * sum_rv - sum_r * sum_v) / (n * sum_r2 - sum_r**2)
a = (sum_v - H * sum_r) / n

print(f"\nSolución del sistema:")
print(f"Constante de Hubble (H) = {H:.2f} km/s/Mpc")
print(f"Intercepto (a) = {a:.2f} km/s")


print("\n(d) GRÁFICO DEL AJUSTE Y DATOS")
print("="*50)

plt.figure(figsize=(10, 6))

# Datos originales
plt.scatter(r, v, color='blue', alpha=0.7, s=60, label='Datos de Hubble')

# Línea de ajuste
r_aj = np.linspace(0, 2.2, 100)
v_aj = a + H * r_aj
plt.plot(r_aj, v_aj, 'r-', linewidth=2, 
         label=f'Ajuste: v = {a:.1f} + {H:.1f}r')

plt.xlabel('Distancia r (Mpc)', fontsize=12)
plt.ylabel('Velocidad v (km/s)', fontsize=12)
plt.title('(d) Ajuste Lineal por Mínimos Cuadrados', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("\n(e) VARIANZA Y DESVIACIONES")
print("="*50)

v_pred = a + H * r
residuales = v - v_pred
varianza_residual = np.sum(residuales**2) / (n - 2)

error_H = np.sqrt(varianza_residual / (np.sum((r - np.mean(r))**2)))

# Error estándar de a
error_a = np.sqrt(varianza_residual * (1/n + np.mean(r)**2 / np.sum((r - np.mean(r))**2)))

varianza = np.var(residuales)
desviacion_estandar = np.std(residuales)


print(f"\nErrores estimados:")
print(f"Error en H = {error_H:.2f} km/s/Mpc")
print(f"Error en a = {error_a:.2f} km/s")

print(f"\nResiduales para cada punto:")
for i in range(len(v)):
    print(f"  Punto {i+1:2d}: v_real = {v[i]:4.0f}, v_pred = {v_pred[i]:5.1f}, residual = {residuales[i]:6.1f}")

chi_cuadrado = np.sum((residuales / sigma_v)**2)


print(f"χ² = {chi_cuadrado:.2f}")
