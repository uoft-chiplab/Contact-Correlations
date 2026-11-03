# -*- coding: utf-8 -*-
"""
@author: Chip Lab

"""
import os
import sys
# Analysis Scripts/
root = os.path.dirname(os.path.dirname(os.getcwd()))

# Fast-Modulation-Contact-Correlation-Project\analysis
analysis_folder = 'f:\\Analysis Scripts\\analysis'
if analysis_folder not in sys.path:
	sys.path.append(analysis_folder)

from data_class import Data
from library import deBroglie, deBroglie_kHz, a97, chi_sq, mK, hbar, h, kB, pi, \
	plt_settings, colors, markers, tintshade, tint_shade_color
from constants import *
from scipy.optimize import curve_fit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

import matplotlib as mpl

def linear(x, m, b):
	return m*x + b

run = Data("2026-11-02_S_e.csv")

df = run.data

df_1ms = df[df['time (ms)'] == 1.0]
df_1ms_avg_amp = df_1ms.groupby('amp (V)').mean(numeric_only = True).reset_index()

df_5ms = df[df['time (ms)'] == 5.0]
df_5ms_avg_amp = df_5ms.groupby('amp (V)').mean(numeric_only = True).reset_index()

fitting_params = []
xs = np.linspace(1,5,10)
for amp in df['amp (V)'].unique():
    x = [1.0,5.0]
    y = [df_1ms_avg_amp[df_1ms_avg_amp['amp (V)'] == amp]['N'].iloc[0],
         df_5ms_avg_amp[df_5ms_avg_amp['amp (V)'] == amp]['N'].iloc[0]]

    popt, pcov = curve_fit(linear, x, y, p0=[1,0])
    
    result = {'amp': amp, 'slope': popt[0], 'intercept': popt[1]}

    fitting_params.append(result)
df_results = pd.DataFrame(fitting_params)

fig, ax = plt.subplots(2,figsize=(14,10))
cmap_color = 'copper'

sc = ax[0].scatter(df_1ms_avg_amp['time (ms)'],df_1ms_avg_amp['N'],
                c=df_1ms_avg_amp['amp (V)'], 
                # s=sizes, 
                cmap=cmap_color, alpha=0.7, edgecolors='none'
                
                )
sc = ax[0].scatter(df_5ms_avg_amp['time (ms)'],df_5ms_avg_amp['N'],
                c=df_5ms_avg_amp['amp (V)'], 
                # s=sizes, 
                cmap=cmap_color, alpha=0.7, edgecolors='none')


cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('amp (V)', fontsize=12)

n_lines = len(df['amp (V)'].unique())
cmap = mpl.colormaps[cmap_color]
colors = cmap(np.linspace(0, 1, n_lines))

for i, amp in enumerate(np.sort(df['amp (V)'].unique())):
	df_results_temp = df_results[df_results['amp'] == amp]
	slope = df_results_temp['slope']
	popt = [slope,df_results_temp['intercept']]
	popt_clean = [float(param.iloc[0]) if hasattr(param, 'iloc') else float(param) for param in popt]
	ax[0].plot(xs, linear(xs, *popt_clean), ls='-',marker='', color = colors[i],
	           label = f'Amp: {amp} V, Slope: {popt_clean[0]:.0f} N/t'
               )

ax[0].set(
	xlabel = 'time (ms)',
	ylabel = 'N'
)
ax[0].legend() 

ax[1].plot(df['cycle'],df['N'], color = colors[4])
ax[1].set(
	ylabel = 'N',
	xlabel = 'cycle'
)

fig.suptitle('Run S 2026-11-02 20kHz')
# fig.tight_layout()