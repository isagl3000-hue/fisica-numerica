#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:10:28 2025

@author: isaias-gl
"""

import numpy as np


def NR(f,df,x0,tol=1e-7,it=100):
    x=x0
    for i in range(it):
        fx=f(x)
        dfx=df(x)
        if abs(dfx)<tol:
            break
        xn=x-fx/dfx
        if abs(xn-x)<tol:
            return xn
        x = xn
    return x

#Inciso (a)
fa = lambda x: np.cos(x)**2 - x**2
dfa = lambda x: -np.sin(2*x) - 2*x

#Inciso (b) 
fb = lambda x: 4*np.cos(x)**2 - x**2
dfb = lambda x: -4*np.sin(2*x) - 2*x

#Inciso (c)
fc = lambda x: 4*np.sin(x)**2 - x**2
dfc = lambda x: 4*np.sin(2*x) - 2*x

sol_a = NR(fa, dfa, 0.7)
sol_b = NR(fb, dfb, 1.5)
sol_c = NR(fc, dfc, 1.9)

print(f"(a) cos²(α₀) = α₀²  →  α₀ ≈ {sol_a:.6f}")
print(f"(b) 4cos²(α₀) = α₀² →  α₀ ≈ {sol_b:.6f}")
print(f"(c) 4sen²(α₁) = α₁² →  α₁ ≈ {sol_c:.6f}")


