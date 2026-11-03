"""This script compares several oscillating field calibrations with different pulse times."""

import sys
import os
import glob

module_folder = 'E:\\Analysis Scripts\\analysis'
if module_folder not in sys.path:
	sys.path.insert(0, 'E:\\Analysis Scripts\\analysis')
from data_class import Data
from fit_functions import Sinc2
from library import *
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# select calibration to use
freq = 6 # kHz
amp = 1.8 # Vpp
period = 1/(freq*1000) # s

###getting the field wiggle calibration done previously 
field_cal_path = os.path.join(os.getcwd(), "field_cal_summary.csv")
field_cal_df = pd.read_csv(field_cal_path)
field_cal_df = field_cal_df[(field_cal_df['wiggle_freq'] == freq) & (field_cal_df['wiggle_amp'] == amp)]
field_cal_df['t/T'] = field_cal_df['pulse_length'] / (period * 1e6)
params = ['B_amp', 'B_phase', 'B_offset']
units = ['(G)', '(rad)', '(G)']
fig, ax = plt.subplots(len(params), 1, figsize=(6, 8), sharex=True)
for i, param in enumerate(params):
	ax[i].errorbar(field_cal_df['t/T'], field_cal_df[param], yerr=field_cal_df['e_'+param], fmt='o', color='cornflowerblue')
	ax[i].set(ylabel=param[2:] + " " + units[i],
		   xlabel = 'Pulse Length over drive period')
fig.suptitle(f"Field Wiggle Calibrations at {freq} kHz, {amp} Vpp")
fig.tight_layout()