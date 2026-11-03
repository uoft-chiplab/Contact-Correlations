# -*- coding: utf-8 -*-
"""
@author: Chip Lab 2024-11-04

This script interpolates trap-averaged contact calculations. 
"""
import numpy as np
import matplotlib.pyplot as plt

ToTFs = np.linspace(0.2, 1.2, 29)

Cs = np.array([2.1783244 , 2.02855915, 1.90539411, 1.78810748, 1.67342239,
       1.56085269, 1.45088512, 1.34443262, 1.24180172, 1.14444157,
       1.05270978, 0.96701767, 0.88750127, 0.81417138, 0.74688656,
       0.68537897, 0.62934843, 0.57842509, 0.53227651, 0.49047896,
       0.45264838, 0.41839089, 0.38739359, 0.35932629, 0.33388573,
       0.31080807, 0.28985089, 0.27077148, 0.25339308])

contact_interpolation = lambda x: np.interp(x, ToTFs, Cs)

if __name__ == "__main__":
	
	# plot interpolation and data
	plt.figure(figsize=(6,4))
	plt.xlabel(r"$T/T_F$")
	plt.ylabel(r"Contact, $C$ [$k_F/N$]")

	plt.plot(ToTFs, Cs, 'o')
	plt.plot(ToTFs, contact_interpolation(ToTFs), '--')
		
	plt.show()