# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:31:36 2024

@author: coldatoms


Tabulated by eye from Vale paper and 
T. Enss, R.Haussmann, W. Zwerger,
Ann.of Phys. 326, 3,2011,770-796,
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
import os

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'theory')


# Thermodynamics of unitary Fermi gas
# Haussmann, Rantner, Cerrito, Zwerger 2007; and
# Enss, Haussmann, Zwerger 2011
# density n=k_F^3/(3\pi^2)
# columns: T/T_F, mu/E_F, u/(E_F*n), s/(k_B*n), p/(E_F*n), C/k_F^4
df = pd.read_csv(os.path.join(data_path,'luttward-thermodyn.txt'),skiprows=4,sep=' ')
test_contact_plot = True
xlabel = 'T/T_F'
ylabel = 'C/k_F^4'
df[ylabel] = df[ylabel] * 3*np.pi**2 # contact density c/(k_F n) = C/k_F^4 * (3 pi^2)
ContactInterpolation = lambda x: np.interp(x, df[xlabel], df[ylabel])
xlow =0
xhigh=1.2
xs = np.linspace(xlow, xhigh, 100)
ylow = 2.2
yhigh = 3.2
if test_contact_plot:
	fig, ax = plt.subplots()
	ax.plot(df[xlabel], df[ylabel], 'ro')
	ax.plot(xs, ContactInterpolation(xs),'r-')
	ax.set_xlabel(r'$T/T_F$')
	ax.set_ylabel(r'$\mathcal{C}/(nk_F)$')
	ax.set_xlim([xlow, xhigh])
	ax.set_ylim([ylow, yhigh])
	ax.set_title('Contact density of UFG, LW tabulation')

# Contact density estimate from Vale paper and Tilman paper
# Crappy eye-balled data
# test_contact_plot = False
# T = np.array([0, 0.05, 0.1, .15, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
# C1 = np.array([3.025, 3.02, 2.98, 2.85, 2.65, 2.58, 2.51, 2.40, 2.29])
# C2 = np.array([0.061, 0.046, 0.038, 0.032, 0.017])	 

# C = np.concatenate((C1, C2*(3*np.pi**2)))

# ContactInterpolation = lambda x: np.interp(x, T, C)
# xs = np.arange(0, 10, 0.1)

# if test_contact_plot:

# 	plt.figure()
# 	plt.plot(T, C, 'ro')
# 	plt.plot(xs, ContactInterpolation(xs), 'r-')
# 	plt.xlabel(r"$T (T_F)$")
# 	plt.ylabel(r'Contact Density $\mathcal{C}/(n k_F)')
# 	# plt.yscale('log')
# 	# plt.xscale('log')
# 	plt.xlim(0,1)
# 	plt.ylim(2, 3.1)
# 	plt.show()