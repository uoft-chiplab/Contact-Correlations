# -*- coding: utf-8 -*-
"""
@author: Chip Lab 2024-11-04

This script interpolates uniform density contact calculations. 
"""
import numpy as np
import matplotlib.pyplot as plt

ToTFs, _, _, _, _, Cdensity = np.loadtxt('luttward-thermodyn.txt', 
										 skiprows=5, unpack=True)

# convert kF^3 to peak trap density of Harmonic trap
# trapped contact density c/(k_F n) = C/k_F^4 * (3 pi^2)
Cs = Cdensity * 3*np.pi**2

ContactInterpolation = lambda x: np.interp(x, ToTFs, Cs)

if __name__ == "__main__":
	
	# plot interpolation and data
	plt.figure(figsize=(6,4))
	plt.xlabel(r"$T/T_F$")
	plt.ylabel(r"Contact density, $\mathcal{C}/(n k_F)$")

	plt.plot(ToTFs, Cs, 'o')
	plt.plot(ToTFs, ContactInterpolation(ToTFs), '--')
		
	plt.show()