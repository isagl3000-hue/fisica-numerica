#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 18:30:28 2025

@author: isaias-gl
"""

import matplotlib.pyplot as plt

# -----------------------------------------------
# 1. Generador de congruencias lineales
# -----------------------------------------------
def lineal_congruential_generator(a, c, M, x0, n):
    """
    Genera n números pseudoaleatorios usando el método de congruencias lineales.
    
    Parámetros:
    a : multiplicador
    c : incremento
    M : módulo
    x0: semilla inicial
    n : cantidad de números a generar
    
    Retorna:
    lista con los n números generados
    """
    sequence = [x0]
    x = x0
    for _ in range(n-1):
        x = (a * x + c) % M
        sequence.append(x)
    return sequence

# -----------------------------------------------
# 2. Parámetros dados
# -----------------------------------------------
a, c, M, x0 = 57, 1, 256, 10
# Generamos más números de los necesarios para asegurar ver el período
n = 300
seq = lineal_congruential_generator(a, c, M, x0, n)

# -----------------------------------------------
# 3. Determinar el período
# -----------------------------------------------
def find_period(sequence):
    """
    Encuentra el período de una secuencia pseudoaleatoria.
    Retorna: (período, índice donde empieza la repetición)
    """
    # Buscamos la primera repetición de la semilla x0
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[0]:
            # Verificamos que realmente sea el período
            periodo = i
            # Comprobamos que los siguientes también coincidan
            ok = True
            for j in range(periodo):
                if j + periodo >= len(sequence):
                    break
                if sequence[j] != sequence[j + periodo]:
                    ok = False
                    break
            if ok:
                return periodo, 0
    return None, None

periodo, start = find_period(seq)
print(f"Semilla x0 = {x0}")
print(f"Período encontrado: {periodo}")
print(f"La secuencia se repite a partir del índice {start}")

# -----------------------------------------------
# 4. Gráfico de pares (x_{2i-1}, x_{2i})
# -----------------------------------------------
# Tomamos los primeros 100 pares para que el gráfico no esté muy saturado
pairs_x = []
pairs_y = []
for i in range(1, min(101, len(seq)//2 + 1)):
    pairs_x.append(seq[2*i-2])
    pairs_y.append(seq[2*i-1])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(pairs_x, pairs_y, alpha=0.6, color='blue')
plt.xlabel(r'$x_{2i-1}$')
plt.ylabel(r'$x_{2i}$')
plt.title('Pares consecutivos (2D)')
plt.grid(True, alpha=0.3)

# -----------------------------------------------
# 5. Gráfico de x_i vs i
# -----------------------------------------------
plt.subplot(1, 2, 2)
plt.plot(range(len(seq)), seq, marker='o', markersize=2, linestyle='-', linewidth=0.5)
plt.xlabel('i (índice)')
plt.ylabel(r'$x_i$')
plt.title('Secuencia completa')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

