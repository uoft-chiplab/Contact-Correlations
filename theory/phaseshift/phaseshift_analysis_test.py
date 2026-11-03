#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Analyse wiggle phase shift measurements from Oct 27 2023, where binding energies and loss amplitude are fit for each time point in the wiggle.

"""
import sys
# module_folder = 'E:\\Analysis Scripts\\analysis'
module_folder = '//Users//kevinxie//Documents//GitHub//analysis//phaseshift'
if module_folder not in sys.path:
	sys.path.insert(0, module_folder)
from data_class import Data
from library import FreqMHz
from fit_functions import Gaussian
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

### load data and set up
# run = '2024-04-05_F'
run = '2024-04-29_F'
run_fn = run + '_e.dat'
meta_df = pd.read_excel('phaseshift_summary.xlsx')
meta_df = meta_df.loc[meta_df['filename'] == run_fn]
run_freq = meta_df.freq.values[0]
x_name = "freq"
y_name = "sum95"
fit_func = Gaussian
guess = [-5000, 43.25, 0.01, 35000] # A, x0, sigma, C
data_folder = 'data/'
run = Data(run_fn, path=data_folder)


### functions ###
# Fixed sinusoidal function depending on given wiggle freq
def FixedSinkHz(t, A, p, C):
	omega = run_freq  * 2 * np.pi # kHz
	return A*np.sin(omega * t - p) + C

# set time to midpoint of pulse
run.data.time = run.data.time + meta_df.pulselength.values[0]/2 / 1000


# # method 1
run.data = run.data[run.data.cyc <200 ]
# agg mean and std for each time, freq pair
df = run.data.groupby(['time','freq']).agg({y_name:['mean','sem']}).reset_index()
# these work only because I know the time, freq pairs are consecutive
df['amp'] = df[y_name, 'mean'] - df[y_name, 'mean'].shift(1)
df['e_amp'] = np.sqrt(df[y_name, 'sem']**2 + df[y_name, 'sem'].shift(1)**2)
# no longer need the bg rows, keep every other row
df = df.iloc[1::2]



#method 2
# run.data = run.data[run.data.cyc < 200]
# bg_mean = run.data[run.data.freq == 42][y_name].mean()
# bg_std = run.data[run.data.freq == 42][y_name].std()
# bg_sem = run.data[run.data.freq==42][y_name].sem()

# df = run.data[run.data.freq != 42].groupby('time').agg({y_name:['mean','sem']}).reset_index()
# df['amp'] = df[y_name,'mean'] - bg_mean
# df['e_amp'] = np.sqrt(df[y_name, 'sem']**2 + bg_sem**2)

# plt.plot(df.time, df.sum95['mean'])
# plt.plot(df.time, df['amp'], 'ro')
# plt.errorbar(df.time, df.amp, df.e_amp)


# method 3
# df = run.data.groupby(['time','freq']).agg({y_name:['mean','std']}).reset_index()
# # these work only because I know the time, freq pairs are consecutive
# df['amp'] = df[y_name, 'mean'] / df[y_name, 'mean'].shift(1)
# df['e_amp'] = np.sqrt((df[y_name, 'std'] / df[y_name, 'mean'])**2 + (df[y_name, 'std'].shift(1) / df[y_name, 'mean'].shift(1))**2)

# # no longer need the bg rows, keep every other row
# df = df.iloc[1::2]


# plt.errorbar(df.time, df.amp, df.e_amp)
# # plt.plot(df.time, df.sum95['mean'], 'ro')
# # plt.plot(df.time, df.amp, 'ro')

xx = np.linspace(0, 1.5, 100)
arb_scale = (df.amp.max()-df.amp.min())/2
phi = 1.85
arb_off = df.amp.mean()
yy = FixedSinkHz(xx, arb_scale, phi, arb_off)

fig, ax = plt.subplots()
ax.plot(df.time, df.amp, 'ro')
ax.plot(xx, yy, 'r--')
ax.errorbar(df.time, df.amp, df.e_amp, ls='none')

bounds = ([0, 0, -np.inf],[np.inf, 2*np.pi, np.inf])
A_popt, A_pcov


arb_scale = (df.sum95['mean'].max() - df.sum95['mean'].min()) / 2
phi = 1.85
arb_off = df.sum95['mean'].mean()
yy2 = FixedSinkHz(xx, arb_scale, phi, arb_off)
fig, ax = plt.subplots()
ax.plot(df.time, df.sum95['mean'], 'ro')
ax.plot(xx, yy2, 'r--')
ax.errorbar(df.time, df.sum95['mean'], df.sum95['sem'], ls='none')

# times = df.time.unique()
# freqs = df.f0.unique()
# amps = df.A.unique()
# e_freqs = df.e_f0.unique()
# e_amps = df.e_A.unique()

# bounds = ([0, 0, -np.inf],[np.inf, 2*np.pi, np.inf])
# f0_guess = [0.1, 1.5, 43.24]
# f0_popt, f0_pcov = curve_fit(FixedSinkHz, df.time.unique(), df.f0.unique(), sigma=df.e_f0.unique(), bounds=bounds, p0=f0_guess)
# A_guess = [100, 2, -8000]
# A_popt, A_pcov = curve_fit(FixedSinkHz, df.time.unique(), df.A.unique(), bounds=bounds, p0=A_guess)
# f0_perr = np.sqrt(np.diag(f0_pcov))
# A_perr = np.sqrt(np.diag(A_pcov))

# xx = np.linspace(0, df.time.max(), 100)
# yyf0 = FixedSinkHz(xx, *f0_popt)
# yyA = FixedSinkHz(xx, *A_popt)

# fig, ax = plt.subplots()
# ax.plot(xx, yyf0, 'b-')
# ax.scatter(df.time.unique(), df.f0.unique(), label = 'f0')
# ax.errorbar(df.time.unique(), df.f0.unique(), yerr = df.e_f0.unique(),ls='none')
# plt.ylabel('f0 [MHz]')
# plt.xlabel('time [us]')
# plt.title('Phase = {:.2f} +/- {:.1f}'.format(f0_popt[1], f0_perr[1]))

# fig, ax = plt.subplots()
# # ax.plot(xx, yyf0, 'b--')
# ax.plot(xx, yyA, 'r-')
# ax.scatter(df.time.unique(), df.A.unique(), label='A')
# ax.errorbar(df.time.unique(), df.A.unique(), yerr=df.e_A.unique(), ls='none')
# plt.ylabel('A')
# plt.xlabel('time [us]')
# plt.title('Phase = {:.2f} +/- {:.1f}'.format(A_popt[1], A_perr[1]))

# print('f0_popt:')
# print(f0_popt, f0_perr)

# print('A_popt:')
# print(A_popt, A_perr)

# ps = A_popt[1] - f0_popt[1]
# e_ps = np.sqrt(A_perr[1]**2 + f0_perr[1]**2)

# print('phase shift: {:.2f} +/- {:.1f}'.format(ps, e_ps))


