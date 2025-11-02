#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 09:34:41 2025

@author: isaias-gl
"""

import numpy as np

#INCISO A. ===================================================================
def original(x):
    return np.sqrt(x + 1) - 1

def reducida(x):
    return x / (np.sqrt(x + 1) + 1)

# Probamos con x cercano a 0
x_values = [1e-10, 1e-15, 1e-20]

print("Comparación de métodos:")
print("x\t\tOriginal\tReducida")
for x in x_values:
    orig = original(x)
    stab = reducida(x)
    print(f"{x:.0e}\t\t{orig:.6e}\t{stab:.6e}")
    
#INCISO B. ===================================================================


def originalB(x, y):
    return np.sin(x) - np.sin(y)

def reducidaB(x, y):
    return 2 * np.cos((x + y) / 2) * np.sin((x - y) / 2)

# Probamos con x ≈ y
x = 1.0
y_values = [1.0 + 1e-10, 1.0 + 5e-15, 1.0 + 1e-20]

print("Comparación de métodos:")
print("y - x\t\tOriginal\t\tReducida")
for y in y_values:
    orig = originalB(x, y)
    stab = reducidaB(x, y)
    print(f"{y-x:.1e}\t\t{orig:.6e}\t{stab:.6e}")
    

#INCISO C. ===================================================================

def originalC(x, y):
    return x**2 - y**2

def reducidaC(x, y):
    return (x - y) * (x + y)

# Probamos con x ≈ y
x = 1.0
y_values = [1.0 + 1e-8, 1.0 + 9e-16, 1.0 + 1e-18]

print("Comparación de métodos:")
print("y - x\t\tOriginal\t\tEstabilizada")
for y in y_values:
    orig = originalC(x, y)
    stab = reducidaC(x, y)
    diff = y - x
    print(f"{diff:.1e}\t\t{orig:.6e}\t{stab:.6e}")

#INCISO D. ===================================================================

def originalD(x):
    return (1 - np.cos(x)) / np.sin(x)

def reducidaD(x):
    return np.tan(x / 2)

# Probamos con x ≈ 0
x_values = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

print("Comparación de métodos:")
print("x\t\tOriginal\t\tReducida\t\tTan(x/2)")
print("-" * 80)
for x in x_values:
    orig = originalD(x)
    stab = reducidaD(x)
    exact = np.tan(x/2)  # Valor de referencia
    
    print(f"{x:.1e}\t\t{orig:.10f}\t{stab:.10f}\t{exact:.10f}")

#INCISO E. ===================================================================

def originalE(a, b, theta):
    return (a**2 + b**2 - 2*a*b*np.cos(theta))**3

def reducidaE(a, b, theta):
    return ((a - b)**2 + 4*a*b*np.sin(theta/2)**2)**3

# Caso 1: a ≈ b, θ pequeño
print("CASO 1: a ≈ b, θ pequeño")
print("=" * 60)
a = 1.0
b = 1.0 + 1e-10
theta = 1e-8

orig = originalE(a, b, theta)
stab = reducidaE(a, b, theta)

print(f"a = {a}, b = {b}, θ = {theta}")
print(f"Diferencia a-b: {b - a:.2e}")
print(f"Original:    {orig:.15e}")
print(f"Estabilizada: {stab:.15e}")
print(f"Diferencia:   {abs(orig - stab):.2e}")
print()












    
