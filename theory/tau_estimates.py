"""This script simply plots the quantum critical timescale \tau as a function of temperature and density (fugacity)."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import os
# automatically read parent dir, assuming pwd is the directory containing this file

# contact correlation proj folder -- Fast-Modulation-Contact-Correlation-Project
root_project = os.path.dirname(os.getcwd())
# analysis folder
root_analysis = os.path.dirname(root_project)
# network machine

root = os.path.dirname(root_analysis)
module_folder = os.path.join(root_analysis, "analysis")
if module_folder not in sys.path:
	sys.path.append(module_folder)

from contact_correlations.UFG_analysis import BulkViscTrap

# Input parameters
EF_est = 16000 # Hz
barnu = 377
num =5 # arbitrary array length
ToTFs = np.linspace(0.25, 0.65, 6)
EFs = np.linspace(10000, 22000, 6)

# plot
fig, axs= plt.subplots(2, figsize=(8,6))
for j in range(len(ToTFs)):
	tau_list = np.array([])
	tauinv_list = np.array([])
	betamutrap_list = np.array([])	
	z_list = np.array([])
	for i in range(len(EFs)):
		EF = EFs[i]
		T = ToTFs[j]*EF
		nus = T*np.logspace(-2, 1, num)
		BVT = BulkViscTrap(ToTFs[j], EF, barnu, nus)
		tau_list = np.append(tau_list, BVT.tau * 1e6 ) # us?
		tauinv_list = np.append(tauinv_list, 1/(BVT.tau * (2*np.pi))/T)
		betamutrap_list = np.append(betamutrap_list, BVT.betamutrap)
		z_list = np.append(z_list, np.exp(BVT.betamutrap))
		#ax.plot(BVT.betamutrap, BVT.tau * 1e6 / (2*np.pi), '-o', label=f'EF={EF}Hz')

	# ax.plot(betamutrap_list, tau_list, 'o', label=f'EF={EF}Hz')
	# ax.set(xlim=[min(betamutrap_list), max(betamutrap_list)],
	# 	   xlabel = r'$\beta\mu$',
	# 	   ylabel = r'$\tau$ [us]')

	axs[0].plot(EFs, tauinv_list, 'o', label=f'ToTF={ToTFs[j]:.2f}, z={z_list[0]:.1f}')
	axs[1].plot(z_list, tauinv_list, 'o', label=f'ToTF={ToTFs[j]:.2f}')

axs[0].legend(loc='best')
axs[0].set(xlim=[min(EFs), max(EFs)],
	   xlabel = r'$E_F$ [Hz]',
	   ylabel = r'$\tau^{-1}/T$')

axs[1].legend(loc='best')
axs[1].set(xlim=[0, 10],
	   xlabel = r'$\exp{\beta \mu}$ ',
	   ylabel = r'$\tau^{-1}/T$')

fig.tight_layout()