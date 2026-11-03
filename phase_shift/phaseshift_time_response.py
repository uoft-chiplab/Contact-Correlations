# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:24:20 2024

@author: Chip Lab
"""

import os
analysis_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '\\analysis'
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
summary = pd.read_excel('./phaseshift/phaseshift_summary.xlsx')
summary = summary[summary['exclude']!=1]
metadata = pd.read_excel('./phaseshift/phaseshift_metadata.xlsx')
df = pd.merge(summary, metadata, how='inner', on='filename')
df['EF_kHz'] = (df['EF_i_kHz']+df['EF_f_kHz'])/2
df['e_EF_kHz'] = np.sqrt(df['EF_i_sem_kHz']**2 + df['EF_f_sem_kHz']**2)
df['ToTF'] =  (df['ToTF_i']+df['ToTF_f'])/2
df['e_ToTF'] = np.sqrt(df['ToTF_i_sem']**2 + df['ToTF_f_sem']**2)
df['kBT'] = df['EF_kHz']*df['ToTF'] # kHz
df['ScaledFreq'] = df['freq']/df['kBT']

# pickle
pickle_file = './time_delay_BVTs.pkl'

load = True

# parameters
ToTFs = [0.2, 
		 0.3, 0.4, 0.5, 0.6]
# EFs = [13e3, 
# 	   14e3, 15e3, 16e3, 18e3]
EFs = [12e3, 14e3, 16e3, 18e3, 20e3]
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
	
	BVTs.append(BVT)
	
with open(pickle_file, 'wb') as f:
	pickle.dump(BVTs, f)


# plot phase shift and time delay of contact response
subfigs = 2

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


for b, BVT in enumerate(BVTs):
	if BVTs[0]:

		pickletime = False
		# compute BVT and save to pickle
		BVTs = []
		analysis = True
		pickle_file = 'time_delay_ToTF_0p2.pkl'
		# analysis loop
		EFs = [12, 14, 16, 18, 20]
		if pickletime == True:
			for i in range(len(EFs)):
				ToTF = 0.2
				EF = EFs[i]
				
				if analysis == False:
					break
				
				T = ToTF*EF
				nus = T*np.logspace(-2, 1, num)
				# compute trap averaged quantities using Tilman's code
				BVT1 = BulkViscTrap(ToTF, EF, barnu, nus)
				
				# compute time delays
				BVT1.time_delay = contact_time_delay(BVT1.phaseshiftsQcrit, 1/BVT1.nus)
				
				BVTs.append(BVT1)

			
			with open(pickle_file, 'wb') as f:
				pickle.dump(BVTs, f)
		else:
			with open(pickle_file, 'rb') as f:
				BVTs = pickle.load(f)
			
		label = r"T={:.0f} kHz".format(BVT.T)
		label2 = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
		axs[1].plot(BVT.nus/BVT.T, BVT.time_delay*1e6, '-', label=label, color=colors[b])
		axs[0].plot(BVT.nus/BVT.T, BVT.phaseshiftsQcrit, '-', label=label2, color=colors[b])
		axs[0].legend()
	
	else:	
		break
		# label = r"T={:.0f} kHz".format(BVT.T)
		# label2 = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
		# axs[1].plot(BVT.nus/BVT.T, BVT.time_delay*1e6, '-', label=label)
		# axs[0].plot(BVT.nus/BVT.T, BVT.phaseshiftsQcrit, '-', label=label2)
		# axs[0].legend()
		# axs[1].legend()

for i in range(len(EFs)):
	if i == len(EFs)-1:
		continue
	plot_df = df[(df['EF_kHz'] > EFs[i]) & (df['EF_kHz'] <= EFs[i+1])]
	axs[0].errorbar(plot_df['ScaledFreq'], plot_df['ps'], yerr=plot_df['e_ps'], \
				 color=colors[i])

plt.tight_layout()
