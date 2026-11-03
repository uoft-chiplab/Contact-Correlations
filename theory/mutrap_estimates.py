# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:48:22 2024

@author: colin
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def linear(x, m, Ei):
	return m*x + Ei


def mutrap_est(ToTF):
	a = -70e3
	b = 36e3
	return a*ToTF + b

Thetas = np.array([0.45, 0.5, 0.55, 0.6, 0.65, 0.7]) # ToTF
mutraps = np.array([5e3, 0, -4e3, -6e3, -9e3, -14e3]) # harmonic trap chemical potentialt as plt

popt, pcov = curve_fit(linear, Thetas, mutraps)

plt.figure()
plt.plot(Thetas, mutraps, 'o')
plt.plot(Thetas, mutrap_est(Thetas), '--')
plt.show()