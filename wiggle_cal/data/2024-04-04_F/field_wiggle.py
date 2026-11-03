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
module_folder = 'E:\\Analysis Scripts\\analysis'
if module_folder not in sys.path:
	sys.path.insert(0, 'E:\\Analysis Scripts\\analysis')
from data_class import Data
from fit_functions import Sinc2, Lorentzian, Gaussian, \
							Parabola, Sin, FixedSin
from scipy.optimize import curve_fit
from library import *
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os

data_folder = os.getcwd()

# run metadata
run = os.path.split(data_folder)[-1] #'yyyy-mm-dd_A'
wiggle_freq = 10.0 # kHz
wiggle_amp = 0.9 # Vpp
pulse_length = 10 # us
note = 'long wait'

# parameters relevent for analysis
regex = re.compile(run + '_e_precharge=(\d+.\d+).dat') # this string probably has to be adjusted for each data set
x_name = "freq"
y_name = "fraction95"
fit_func = Sinc2
num = 500
save_final_plot = True

# Fixed sinusoidal function depending on given wiggle freq
def FixedSinkHz(t, A, p, C):
	omega = wiggle_freq/1000.0 * 2 * np.pi # kHz
	return A*np.sin(omega*t - p) + C

# initalize no guesses, but fill them in if needed
guess = [0.05, 2, 202.1]

# if this is wrong then I don't wanna be right
fn_list = []
time_list = []
popt_list = []
perr_list = []
B_list = []
e_B_list = []

### Fit 97 transfer scans as function of frequency
for file in os.listdir(data_folder):
	res = regex.match(file)
	if not res:
		 continue
	    
	# read data
	data = Data(file, path=data_folder)
	fn_list.append(file)
	
	# get time of pulse (calculation may depend on how run was set up)
	delaytime = data.data['delaytime'][0]
	time = delaytime*1000 + (pulse_length/2) # choose middle of pulse
	time_list.append(time)
	
	# fit fraction95 vs. freq with Sinc2
	data.fit(fit_func, names = [x_name, y_name])
	popt_list.append(data.popt)
	perr_list.append(data.perr)
	
	# convert peak freq to B field
	data.B = B_from_FreqMHz(data.popt[1])
	data.Berr = np.abs(data.B-B_from_FreqMHz(data.popt[1]+data.perr[1]))
	B_list.append(data.B)
	e_B_list.append(data.Berr)
	
# create summary dataframe of analysis and export
data_dict = {'filename':fn_list, 'time':time_list, 'popt':popt_list, 'perr':perr_list, 'B':B_list, 'e_B':e_B_list}	
data_df = pd.DataFrame.from_dict(data_dict)
# Extract elements of popt and perr lists into their own columns
split_df1 = pd.DataFrame(data_df['popt'].tolist(), columns=['A','x0','sigma','C'])
split_df2 = pd.DataFrame(data_df['perr'].tolist(), columns = ['e_A','e_x0','e_sigma','e_C'])
data_df = pd.concat([data_df, split_df1], axis=1)
data_df = pd.concat([data_df, split_df2], axis=1)
data_df.drop(['popt','perr'],axis=1, inplace=True)
data_df.to_csv(run+'_analysis_summary.csv')

print('fitting B vs. time....')
# fit B vs. time
func=FixedSinkHz
param_bounds = [[0, 0, -np.inf],[np.inf, 2*np.pi, np.inf]]
popt, pcov = curve_fit(func, data_df['time'], data_df['B'], sigma=data_df['e_B'], bounds=param_bounds, p0=guess)
perr = np.sqrt(np.diag(pcov))

fit_params=['Amplitude','Phase','Offset']
parameter_table = tabulate([['Values', *popt], 
								 ['Errors', *perr]], 
								 headers=fit_params)
print("Field calibration:")
print("")
print(parameter_table)
	
# stuff result into summary csv file
fcal_dict = {'run':run, 'wiggle_freq':wiggle_freq, 'wiggle_amp':wiggle_amp, 'B_amp':popt[0], 'B_phase':popt[1], 'B_offset':popt[2],\
			 'e_B_amp':perr[0], 'e_B_phase':perr[1], 'e_B_offset':perr[2], 'note':note}
fcal_df = pd.DataFrame.from_dict([fcal_dict])

summ_path = 'E:\\Analysis Scripts\\analysis\\data\\FieldWiggleCal\\field_cal_summary.csv'
summ_df = pd.read_csv(summ_path)

if summ_df['run'].str.contains(run).any():
	idx = summ_df.index[summ_df['run']==run]
	summ_df.drop(idx,inplace=True)
	update_df = pd.concat([summ_df, fcal_df])
else :
	update_df = pd.concat([summ_df, fcal_df])

pd.DataFrame.to_csv(update_df, summ_path, index=False)

# plot B vs. time
plt.figure()
xx = np.linspace(np.min(data_df['time']),
				   np.max(data_df['time']), num)
plt.plot(xx, func(xx, *popt), "--")
plt.errorbar(data_df['time'],data_df['B'], 
			 yerr=data_df['e_B'], fmt='go')
plt.title(run + ' ' + str(wiggle_freq) + " kHz, " + str(wiggle_amp) + " Vpp field wiggle cal")
plt.xlabel('time [us]')
plt.ylabel('field [G]')
if save_final_plot:
	plt.savefig(run + ' ' + str(wiggle_freq) + ' kHz, ' + str(wiggle_amp) + ' Vpp field wiggle cal.png')
	
plt.show()