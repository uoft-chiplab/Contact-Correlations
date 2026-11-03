# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:24:20 2024

@author: Chip Lab
"""

import os
#analysis_folder = 'E:\\\\Analysis Scripts\\analysis\\'
analysis_folder = '\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\analysis'
import sys
if analysis_folder not in sys.path:
	sys.path.append(analysis_folder)
from library import styles, colors
from contact_correlations.UFG_analysis import BulkViscTrap
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# summary of phase shift measurements
summary = pd.read_excel(analysis_folder + '\\contact_correlations\\phaseshift\\phaseshift_summary.xlsx')
summary = summary[summary['exclude']!=1]
metadata = pd.read_excel(analysis_folder + '\\contact_correlations\\phaseshift\\phaseshift_metadata.xlsx')
df = pd.merge(summary, metadata, how='inner', on='filename')
df['EF_kHz'] = (df['EF_i_kHz']+df['EF_f_kHz'])/2
df['e_EF_kHz'] = np.sqrt(df['EF_i_sem_kHz']**2 + df['EF_f_sem_kHz']**2)
df['ToTF'] =  (df['ToTF_i']+df['ToTF_f'])/2
df['e_ToTF'] = np.sqrt(df['ToTF_i_sem']**2 + df['ToTF_f_sem']**2)
df['kBT'] = df['EF_kHz']*df['ToTF'] # kHz
df['ScaledFreq'] = df['freq']/df['kBT']
df['time_lag'] = df['ps']/df['freq']/2/np.pi
df['e_time_lag'] = df['e_ps']/df['freq']/2/np.pi

# pickle
pickle_file = analysis_folder + '\\contact_correlations\\time_delay_BVTs_working.pkl'
load = True

# parameters
ToTFs = np.array(df['ToTF'])
EFs = np.array(df['EF_kHz']*1000)
barnu = 377
num = 100

def contact_time_delay(phi, period):
	""" Computes the time delay of the contact response given the oscillation
		period and the phase shift.
	"""
	return phi/(2*np.pi) * period

# load pickle
if load == True:
	with open(pickle_file, 'rb') as f:
		BVTs = pickle.load(f)
	
	analysis = False
		
else: 
	# compute BVT and save to pickle
	BVTs = []
	analysis = True

# analysis loop
for i in range(len(ToTFs)):
	ToTF = ToTFs[i]
	EF = EFs[i]
	
	if analysis == False:
		break
	
	T = ToTF*EF
	nus = T*np.logspace(-2, 1, num)
	# compute trap averaged quantities using Tilman's code
	BVT = BulkViscTrap(ToTF, EF, barnu, nus)
	# compute time delays
	BVT.time_delay = contact_time_delay(BVT.phaseshiftsQcrit, 1/BVT.nus)
	BVT.time_delay_LR = contact_time_delay(BVT.phiLR, 1/BVT.nus)
	BVTs.append(BVT)

with open(pickle_file, 'wb') as f:
	pickle.dump(BVTs, f)
# plot phase shift and time delay of contact response
subfigs = 3
fig, axes = plt.subplots(1, subfigs, figsize=(5*subfigs, 4))
axs = axes.flatten()
# phase shift
ax = axs[0]
ax.set(xlabel=r"Frequency, $h\nu/k_BT$", 
	   ylabel=r"Phase Shift, $\phi$ [rad]", xscale='log',
	   ylim=[0,1])
# time delay
ax = axs[1]
ax.set(xlabel=r"Frequency, $h\nu/k_BT$", 
	   ylabel=r"Response Delay, $t_{delay}$ [us]", xscale='log')
# rescaled time delay
ax=axs[2]
ax.set(xlabel=r"Frequency, $h\nu/k_BT$",
	   ylabel=r"Rescaled Response Delay, $t_{delay}/\tau$", xscale='log')

for b, BVT in enumerate(BVTs):
		BVT.time_delay_LR = contact_time_delay(BVT.phiLR, 1/BVT.nus)	
		label = r"T={:.0f} kHz".format(BVT.T)
		label2 = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
		#phitest =  np.arctan(2*np.pi*BVT.nus * BVT.tau / (1 + (2*np.pi*BVT.nus*BVT.tau)**2))
		axs[0].plot(BVT.nus/BVT.T, BVT.phaseshiftsQcrit, '-', label=label2, color=colors[b])
		axs[0].plot(BVT.nus/BVT.T, BVT.phiLR, ':', color=colors[b])
		axs[1].plot(BVT.nus/BVT.T, BVT.time_delay*1e6, '.', label=label+' HD', color=colors[b])
		axs[1].plot(BVT.nus/BVT.T, BVT.time_delay_LR*1e6, ':', label=label+' LR', color=colors[b])
		axs[2].plot(BVT.nus/BVT.T, BVT.time_delay/BVT.tau, '-', label=label+' HD',color=colors[b])
		axs[2].plot(BVT.nus/BVT.T, BVT.time_delay_LR/BVT.tau, ':', label = label+ ' LR',color=colors[b])
		plot_df = df[(df['EF_kHz'] == EFs[b]/1000)]
		axs[1].errorbar(plot_df['ScaledFreq'], 
				  plot_df['time_lag']*1e3,
				  yerr=plot_df['e_time_lag']*1e3, color=colors[b])
		axs[0].errorbar(plot_df['ScaledFreq'], plot_df['ps'], yerr=plot_df['e_ps'], \
				  color=colors[b])
		axs[2].errorbar(plot_df['ScaledFreq'], 
				  plot_df['time_lag']/1e3/BVT.tau,
				  yerr=plot_df['e_time_lag']/1e3/BVT.tau, color=colors[b])
			
		axs[2].set(ylim=[-0.1,1.1],yscale='linear')
		axs[1].set(ylim=[-5, 40], yscale='linear')
		axs[0].set(ylim=[0, np.pi/2])
		axs[0].legend()
	# else:	
	# 	break
# 		label = r"T={:.0f} kHz".format(BVT.T)
# 		label2 = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
# 		axs[1].plot(BVT.nus/BVT.T, BVT.time_delay*1e6, '-', label=label)
# 		axs[0].plot(BVT.nus/BVT.T, BVT.phaseshiftsQcrit, '-', label=label2)
# 		axs[0].legend()
# 		axs[1].legend()

# for i in range(len(EFs)):
# 	if i == len(EFs)-1:
# 		continue
# 	plot_df = df[(df['EF_kHz'] > EFs[i]) & (df['EF_kHz'] <= EFs[i+1])]
# 	axs[0].errorbar(plot_df['ScaledFreq'], plot_df['ps'], yerr=plot_df['e_ps'], \
# 				 color=colors[i])

# plt.tight_layout()
# fig, ax= plt.subplots()
# phitest =  np.arctan(BVT.nus * BVT.tau / (1 + (BVT.nus*BVT.tau)**2))
# ax.plot(BVT.nus/BVT.T, phitest, '-')
# ax.set(xscale='log')