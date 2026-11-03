# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:26:57 2024

@author: coldatoms
"""

import os
analysis_folder = 'E:\\\\Analysis Scripts\\analysis\\'
import sys
if analysis_folder not in sys.path:
	sys.path.append(analysis_folder)
import re
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from library import FreqMHz, h
from data_class import Data
import matplotlib.colors as mc
import colorsys

summary = pd.read_excel('./phaseshift_summary.xlsx')
summary = summary[summary['exclude']!=1]

metadata = pd.read_excel('./phaseshift_metadata.xlsx')
df = pd.merge(summary, metadata, how='inner', on='filename')

df['EF_kHz'] = (df['EF_i_kHz']+df['EF_f_kHz'])/2
df['e_EF_kHz'] = np.sqrt(df['EF_i_sem_kHz']**2 + df['EF_f_sem_kHz']**2)
df['ToTF'] =  (df['ToTF_i']+df['ToTF_f'])/2
df['e_ToTF'] = np.sqrt(df['ToTF_i_sem']**2 + df['ToTF_f_sem']**2)
df['kBT'] = df['EF_kHz']*df['ToTF'] # kHz
df['ScaledFreq'] = df['freq']/df['kBT']
fig, ax = plt.subplots()

ax.errorbar(df['ScaledFreq'], df['ps'], yerr=df['e_ps'])
ax.set(xlabel=r'Frequency, $h\nu/k_B T$', ylabel = 'Phase shift [rad]')

# ax=axs[1]
# ax.plot(summary['freq'], summary['Amp'])