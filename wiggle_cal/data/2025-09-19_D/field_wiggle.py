# -*- coding: utf-8 -*-
"""
field_wiggle.py
2023-11-14
@author: Chip Lab

Analysis field wiggle scans where the frequency of transfer is 
varied, and the delay time is varied. Files are organized by
delay time due to how MatLab outputs the .dat files.
"""
import sys
import os
import glob

module_folder = 'E:\\Analysis Scripts\\analysis'
if module_folder not in sys.path:
	sys.path.insert(0, 'E:\\Analysis Scripts\\analysis')
from data_class import Data
from fit_functions import Sinc2, Lorentzian, Gaussian, \
							Parabola, Sin, FixedSin
from library import *
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import convolve

data_folder = os.getcwd()

# run metadata
run = os.path.split(data_folder)[-1] #'yyyy-mm-dd_A
wiggle_freq = 6 # kHz
wiggle_amp = 1.8 # Vpp
note = ''

# parameters relevent for analysis
x_name = "freq (MHz)"
y_name = "fraction95"
fit_func = Sinc2
num = 500
save_final_plot = True

# Fixed sinusoidal function depending on given wiggle freq
def FixedSinkHz(t, A, p, C):
	omega = wiggle_freq/1000.0 * 2 * np.pi # kHz
	return A*np.sin(omega*t - p) + C

# initalize no guesses, but fill them in if needed
guess = [0.035, 2, 202.16]

file_list = glob.glob(f"{os.path.dirname(__file__)}\\*pre*.dat")
times = [float(path.split("=")[-1][:-4]) for path in file_list]

data_df = pd.DataFrame({})

for file in file_list:
	data = Data(file, path=data_folder)
	data_df = pd.concat([data_df, data.data], ignore_index=True)

pulse_lengths = np.unique(data_df["pulse time"])*1e3 # ms to us
	
popt_list = []
perr_list = []
B_list = []
e_B_list = []
delay_time_list = []
pulse_length_list = []

data_df['pulse_length'] = data_df['pulse time'] * 1e3  # Convert ms to us
data_df['time'] = data_df['wiggle time pre']*1000 + data_df['pulse_length']/2.0

### Fit 97 transfer scans as function of frequency
for i, time in enumerate(data_df.time.unique()):
	for j, pulse_length in enumerate(pulse_lengths):
	
		this_df = data_df[(data_df.time == time) & (data_df.pulse_length == pulse_length)]
		if this_df.empty:
			continue

		delay_time_list.append(time)
		pulse_length_list.append(pulse_length)
		
		# This is dumb, we're making a Data class using the last file, but just overwriting the data
		data = Data(file, path=data_folder)
		data.data = this_df
		data.filename = f'delay time = {time} us, pulse length = {pulse_length} us'
		
		# fit fraction95 vs. freq with Sinc2
		data.fit(fit_func, names = [x_name, y_name])
		popt_list.append(data.popt)
		perr_list.append(data.perr)
		
		# convert peak freq to B field
		data.B = B_from_FreqMHz(data.popt[1])
		data.Berr = np.abs(data.B - B_from_FreqMHz(data.popt[1] + data.perr[1]))
		B_list.append(data.B)
		e_B_list.append(data.Berr)
	
# create summary dataframe of analysis and export
data_dict = {'time':delay_time_list, 'pulse_length':pulse_length_list,
			 'popt':popt_list, 'perr':perr_list, 'B':B_list, 'e_B':e_B_list}	
df = pd.DataFrame.from_dict(data_dict)

# Initialize plot
plt.figure()
ax = plt.subplot()
# ax.set(ylim=[202.11, 202.2])
ax.yaxis.get_major_formatter().set_useOffset(False)

# Loop over different pulse times
for pulse_length in pulse_lengths:
	fit_df = df[df.pulse_length == pulse_length].reset_index()

	# Extract elements of popt and perr lists into their own columns
	split_df1 = pd.DataFrame(fit_df['popt'].tolist(), columns=['A','x0','sigma','C'])
	split_df2 = pd.DataFrame(fit_df['perr'].tolist(), columns = ['e_A','e_x0','e_sigma','e_C'])
	fit_df = pd.concat([fit_df, split_df1], axis=1)
	fit_df = pd.concat([fit_df, split_df2], axis=1)
	fit_df.drop(['popt','perr'], axis=1, inplace=True)
	fit_df.to_csv(run + f'_{pulse_length}us_pulse_length_analysis_summary.csv')

	print('fitting B vs. time....')
	# fit B vs. time
	func=FixedSinkHz
	param_bounds = [[0, 0, -np.inf],[np.inf, 2*np.pi, np.inf]]
	popt, pcov = curve_fit(func, fit_df['time'], fit_df['B'], 
						sigma=fit_df['e_B'], 
						bounds=param_bounds, p0=guess)
	perr = np.sqrt(np.diag(pcov))

	fit_params=['Amplitude','Phase','Offset']
	parameter_table = tabulate([['Values', *popt], 
									['Errors', *perr]], 
									headers=fit_params)
	print("Field calibration:")
	print("")
	print(parameter_table)
		
	# stuff result into summary csv file
	fcal_dict = {'run':run, 'pulse_length':pulse_length, 'wiggle_freq':wiggle_freq, 'wiggle_amp':wiggle_amp, 
			  	'B_amp':popt[0], 'B_phase':popt[1], 'B_offset':popt[2],
				'e_B_amp':perr[0], 'e_B_phase':perr[1], 'e_B_offset':perr[2], 'note':note}
	fcal_df = pd.DataFrame.from_dict([fcal_dict])

	summ_path = 'E:\\Analysis Scripts\\analysis\\contact_correlations\\FieldWiggleCal\\field_cal_summary.csv'
	summ_df = pd.read_csv(summ_path)

	if summ_df['run'].str.contains(run).any():
		idx = summ_df.index[summ_df['run']==run]
		summ_df.drop(idx,inplace=True)
		update_df = pd.concat([summ_df, fcal_df])
	else :
		update_df = pd.concat([summ_df, fcal_df])

	pd.DataFrame.to_csv(update_df, summ_path, index=False)

	# plot B vs. time
	xx = np.linspace(np.min(fit_df['time']),
					np.max(fit_df['time']), num)
	ax.plot(xx, func(xx, *popt), "--", color="orchid",
		 label=f'fit [{popt[0]:.3f}({perr[0]*1e3:.0f}), {popt[1]:.2f}({perr[1]*1e2:.0f}), {popt[2]:.3f}({perr[2]*1e3:.0f})]')
	# ax.plot(xx, func(xx, *[popt[0], popt[1]+0.8, popt[2]]), "-.", color='orange', label=f'exp response (0.8 rad delay)')
	ax.errorbar(fit_df['time'],fit_df['B'], 
				yerr=fit_df['e_B'], fmt='o', label=f'{pulse_length} us', color="mediumvioletred")
	

ax.set(title=run[0:10]+ ' ' + str(wiggle_freq) + " kHz, " + str(wiggle_amp) + " Vpp field wiggle cal")
ax.set(xlabel='Time [us]', ylabel='Field [G]')

if save_final_plot:
	plt.savefig(run + ' ' + str(wiggle_freq) + ' kHz, ' + str(wiggle_amp) + ' Vpp field wiggle cal.png')
	
ax.legend()
plt.show()

# convolve sine and sinc^2