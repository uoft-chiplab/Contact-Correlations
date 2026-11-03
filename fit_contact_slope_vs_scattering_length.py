# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:35:28 2024

@author: coldatoms
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from fit_functions import Linear
from library import *

linear, names, guess = Linear([])

B = np.array([201.9, 202.1, 202.3])
eB = np.array([0.01, 0.01, 0.01])
C = np.array([1.27, 0.78, 0.444])
eC = np.array([0.02, 0.01, 0.009])

EF = h*22000 # 22kHz
kF = np.sqrt(2*mK*EF)/hbar

inv_kFa97 = 1/a97(B)/kF
e_inv_kFa97 = np.abs(inv_kFa97-1/a97(B+eB)/kF)

plt.figure()
xlabel = "1/(kF a)"
ylabel = "C/(kF N)"
plt.xlabel(xlabel)
plt.ylabel(ylabel)

popt,pcov = curve_fit(linear, inv_kFa97, C)
perr = np.sqrt(np.diag(pcov))
num = 100
xlist = np.linspace(inv_kFa97[0], inv_kFa97[-1], num)

plt.plot(xlist, linear(xlist, *popt), '-')
plt.errorbar(inv_kFa97, C, yerr=eC, fmt='o--')

plt.show()