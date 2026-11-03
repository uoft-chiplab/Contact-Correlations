#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Analyse wiggle phase shift measurements from Oct 27 2023, where binding energies and loss amplitude are fit for each time point in the wiggle.

"""
import sys
import os
module_folder = 'E:\\Analysis Scripts\\analysis'
if module_folder not in sys.path:
	sys.path.insert(0, 'E:\\Analysis Scripts\\analysis')
from data_class import Data
from library import FreqMHz
from fit_functions import Gaussian, Sinc2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

### load data and set up
run_fn = '2024-04-29_F_e.dat'
# run_fn = '2025-09-24_D_e.dat'
meta_df = pd.read_excel('phaseshift_summary.xlsx')
meta_df = meta_df.loc[meta_df['filename'] == run_fn[:-4]]
run_freq = meta_df.freq.values[0]
x_name = "freq"
y_name = "sum95"
fit_func = Gaussian
guess = [-5000, 43.25, 0.01, 35000] # A, x0, sigma, C
data_folder = os.path.dirname(__file__) +'\phaseshift\data'
run = Data(run_fn, path=data_folder)


### functions ###
# Fixed sinusoidal function depending on given wiggle freq
def FixedSinkHz(t, A, p, C):
	omega = run_freq / 1000 * 2 * np.pi # kHz
	return A*np.sin(omega * t - p) + C

Bcal_fn = meta_df.Bcal_run.values[0]
Bcal_df = pd.read_csv(os.path.dirname(__file__) +'/FieldWiggleCal/field_cal_summary.csv')
Bcal_df = Bcal_df.loc[Bcal_df.run == Bcal_fn]
def Bfield_from_time(t, arb_scaling=1):
    omega=2*np.pi*Bcal_df.wiggle_freq.values[0] / 1000 # kHz
    amp= Bcal_df.B_amp#0.0695243 #+/- 0.0025798
    phase= Bcal_df.B_phase#2.17306 #+/- 0.0474719
    offset= Bcal_df.B_offset#202.072 #+/- 0.00203875
    return arb_scaling * amp * np.sin(omega * t + phase) + offset

def Eb_from_field(B, whichfit='linear'):
    '''
    Eb cal from Oct 30
    Eb in MHz, field input in G
    '''
    if whichfit == 'linear':
        a=-0.05124 #+/- 0.006695
        b=14.35 #+/- 1.353
        return a*B + b

### switches ###
# FIX_EB=False
# FIX_WIDTH=False
# fixedwidth=0.034542573613013806 #taken from the median sigma of an unfixed result
# save_intermediate=False
# save_final=False
# if FIX_EB and FIX_WIDTH:
#     suffix_str = 'f0 fixed and sigma fixed'
# elif FIX_EB:
#     suffix_str = 'f0 fixed'
# elif FIX_WIDTH:
#     suffix_str = 'sigma fixed'
# else:
#     suffix_str = 'unfixed'
    
subrun_list = []
### Analysis loop
for time in run.data.time.unique():
    # not efficient but make new object and then filter
    subrun = Data(run_fn, path=data_folder)
    subrun.data = subrun.data.loc[subrun.data.time == time]
    print(meta_df.pulselength)
    subrun.data.time = subrun.data.time * 1000 + (meta_df.pulselength.values[0]/2.0) # ms->us + 1/2length, won't be consistent across all data
    subrun.fit(fit_func, names = [x_name, y_name], guess=guess, label=str(subrun.data.time.iloc[0])+' us')
    
    [subrun.data['A'], subrun.data['f0'], subrun.data['sigma'], subrun.data['C']] = \
        [*subrun.popt]
        
    [subrun.data['e_A'], subrun.data['e_f0'], subrun.data['e_sigma'], subrun.data['e_C']] = \
        [*subrun.perr]
    
    subrun_list.append(subrun.data)
   
df = pd.concat(subrun_list)
df['field'] = Bfield_from_time(df['time'])
df['Eb'] = Eb_from_field(df['field'])

skip_time = np.array([0.150]) * 1000 + meta_df.pulselength.values[0]/2
df = df[~df.time.isin(skip_time)]
times = df.time.unique()
freqs = df.f0.unique()
amps = df.A.unique()
e_freqs = df.e_f0.unique()
e_amps = df.e_A.unique()

bounds = ([0, 0, -np.inf],[np.inf, 2*np.pi, np.inf])
f0_guess = [0.1, 1.5, 43.24]
f0_popt, f0_pcov = curve_fit(FixedSinkHz, df.time.unique(), df.f0.unique(), sigma=df.e_f0.unique(), bounds=bounds, p0=f0_guess)
A_guess = [100, 2, -8000]
A_popt, A_pcov = curve_fit(FixedSinkHz, df.time.unique(), df.A.unique(), bounds=bounds, p0=A_guess)
f0_perr = np.sqrt(np.diag(f0_pcov))
A_perr = np.sqrt(np.diag(A_pcov))

xx = np.linspace(0, df.time.max(), 100)
yyf0 = FixedSinkHz(xx, *f0_popt)
yyA = FixedSinkHz(xx, *A_popt)

fig, ax = plt.subplots()
ax.plot(xx, yyf0, 'b-')
ax.scatter(df.time.unique(), df.f0.unique(), label = 'f0')
ax.errorbar(df.time.unique(), df.f0.unique(), yerr = df.e_f0.unique(),ls='none')
plt.ylabel('f0 [MHz]')
plt.xlabel('time [us]')
plt.title('Phase = {:.2f} +/- {:.1f}'.format(f0_popt[1], f0_perr[1]))

fig, ax = plt.subplots()
# ax.plot(xx, yyf0, 'b--')
ax.plot(xx, yyA, 'r-')
ax.scatter(df.time.unique(), df.A.unique(), label='A')
ax.errorbar(df.time.unique(), df.A.unique(), yerr=df.e_A.unique(), ls='none')
plt.ylabel('A')
plt.xlabel('time [us]')
plt.title('Phase = {:.2f} +/- {:.1f}'.format(A_popt[1], A_perr[1]))

print('f0_popt:')
print(f0_popt, f0_perr)

print('A_popt:')
print(A_popt, A_perr)

ps = A_popt[1] - f0_popt[1]
e_ps = np.sqrt(A_perr[1]**2 + f0_perr[1]**2)

print('phase shift: {:.2f} +/- {:.1f}'.format(ps, e_ps))


