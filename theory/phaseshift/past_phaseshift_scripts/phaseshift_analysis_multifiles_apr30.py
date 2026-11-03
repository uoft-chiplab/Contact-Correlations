#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Analyse wiggle phase shift measurements from Oct 27 2023, where binding energies and loss amplitude are fit for each time point in the wiggle.

"""
import re
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from fit_functions import Gaussian, Sinc2
from library import FreqMHz
from data_class import Data
import sys
# module_folder = 'E:\\Analysis Scripts\\analysis'
module_folder = '//Users//kevinxie//Documents//GitHub//analysis//phaseshift'
if module_folder not in sys.path:
    sys.path.insert(0, module_folder)

# load data and set up
# run = '2024-04-05_F'
# run = '2024-04-29_H'
run = '2024-04-30_K'
# run = '2024-05-02_B'
boxsize = 'midbox'
run_fn = run + '_e_' + boxsize+'_time=0.(\d+).dat'
meta_df = pd.read_excel('phaseshift_summary.xlsx')
meta_df = meta_df.loc[meta_df['filename'] == '2024-04-30_K_e.dat']
run_freq = meta_df.freq.values[0]
x_name = "freq"
# y_name = "two2D_int1"
y_name = 'c5'
fit_func = Gaussian
guess = [-5000, 43.2, 0.02, 30000]  # A, x0, sigma, C
data_folder = 'data/'
# run = Data(run_fn, path=data_folder)
regex = re.compile(run_fn)
# regex2 = re.compile('2024-05-02_B_e_sig1_time=0.(\d+).dat')

### functions ###
# Fixed sinusoidal function depending on given wiggle freq


def FixedSinkHz(t, A, p, C):
    omega = run_freq / 1000 * 2 * np.pi  # kHz
    return A*np.sin(omega * t - p) + C


subrun_list = []
for file in os.listdir(data_folder):
    res = rege1.match(file)
    # res2 = regex2.match(file)
    if not res:
        continue
    # if res1:
    #     time_mod = 200
    #     continue
    # elif res2:
    #     time_mod = 0
    # else:
    #     continue

    subrun = Data(file, path=data_folder)
    # if file == run + '_e_time=0.11.dat':
    #     subrun.data.time = -0.09
    # elif file == run+ '_e_time=0.06.dat':
    #     subrun.data.time = -0.14
    # elif file == run+ '_e_time=0.01.dat':
    #     subrun.data.time = -0.19

    # ms->us + 1/2length, won't be consistent across all data
    subrun.data.time = subrun.data.time * 1000 + \
        (meta_df.pulselength.values[0]/2.0) + time_mod
    # if (subrun.data.time.values[0] == 220 or subrun.data.time.values[0] == 270) and res1:
        # continue
    subrun.fit(fit_func, names=[x_name, y_name], guess=guess, label=str(
        subrun.data.time.iloc[0])+' us')

    [subrun.data['A'], subrun.data['f0'], subrun.data['sigma'], subrun.data['C']] = \
        [*subrun.popt]

    [subrun.data['e_A'], subrun.data['e_f0'], subrun.data['e_sigma'], subrun.data['e_C']] = \
        [*subrun.perr]
    
    subrun.data['Aosum'] = subrun.data['A'] / subrun.data['sum95'].mean()
    subrun.data['AoC'] = subrun.data['A'] / subrun.data['C']
    subrun.data['e_Aosum']= np.abs(subrun.data['Aosum'] * \
        np.sqrt((subrun.data['e_A']/subrun.data['A'])**2 + (subrun.data['sum95'].std()/subrun.data['sum95'].mean())**2  ))
    subrun.data['e_AoC']= np.abs(subrun.data['AoC'] * \
        np.sqrt((subrun.data['e_A']/subrun.data['A'])**2 + (subrun.data['e_C']/subrun.data['C'])**2  ))
    subrun_list.append(subrun.data)

df = pd.concat(subrun_list)

# the result from this point looked weird. I think there are issues at long time
# skip_time = np.array([0.08,0.18,0.28,0.38,0.48]) * \
    # 1000 + meta_df.pulselength.values[0]/2
# df = df[~df.time.isin(skip_time)]
times = df.time.unique()
freqs = df.f0.unique()
amps = df.A.unique()
sigmas = np.abs(df.sigma.unique())
e_freqs = df.e_f0.unique()
e_amps = df.e_A.unique()
e_sigmas = df.e_sigma.unique()

fig, ax = plt.subplots()
ax.plot(times, sigmas, 'go', label='sigma')
ax.errorbar(times, sigmas, yerr=e_sigmas, ls='none')
plt.show()

bounds = ([0, 0, -np.inf], [np.inf, 2*np.pi, np.inf])  # amp, phase, offset
f0_guess = [0.1, 1.5, 43.20]
f0_popt, f0_pcov = curve_fit(FixedSinkHz, df.time.unique(
), df.f0.unique(), sigma=df.e_f0.unique(), bounds=bounds, p0=f0_guess)
A_guess = [1000, 2, -8000]
A_popt, A_pcov = curve_fit(FixedSinkHz, df.time.unique(
), df.A.unique(), sigma=df.e_A.unique(), bounds=bounds, p0=A_guess)

Anorm_guess = [0.2, 2, -0.5]
Anorm_name = 'AoC'
Anorm_popt, Anorm_pcov = curve_fit(FixedSinkHz, df.time.unique(
), df[Anorm_name].unique(), bounds=bounds, p0=Anorm_guess, sigma=df['e_'+Anorm_name].unique())

f0_perr = np.sqrt(np.diag(f0_pcov))
A_perr = np.sqrt(np.diag(A_pcov))
Anorm_perr = np.sqrt(np.diag(Anorm_pcov))

xx = np.linspace(df.time.min(), df.time.max(), 100)
yyf0 = FixedSinkHz(xx, *f0_popt)
yyA = FixedSinkHz(xx, *A_popt)
yyAnorm = FixedSinkHz(xx, *Anorm_popt)

fig, ax = plt.subplots()
ax.plot(xx, yyf0, 'b-')
ax.scatter(df.time.unique(), df.f0.unique(), label='f0')
ax.errorbar(df.time.unique(), df.f0.unique(), yerr=df.e_f0.unique(), ls='none')
plt.ylabel('f0 [MHz]')
plt.xlabel('time [us]')
title_str = 'Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    f0_popt[1], f0_perr[1], y_name, fit_func.__name__, boxsize)
plt.title(title_str)
plt.savefig('Peak frequency.png', dpi=300)

fig, ax = plt.subplots()
# ax.plot(xx, yyf0, 'b--')
ax.plot(xx, yyA, 'r-')
ax.scatter(df.time.unique(), df.A.unique(), label='A')
ax.errorbar(df.time.unique(), df.A.unique(), yerr=df.e_A.unique(), ls='none')
plt.ylabel('A')
plt.xlabel('time [us]')
title_str = 'Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    A_popt[1], A_perr[1], y_name, fit_func.__name__, boxsize)
plt.title(title_str)
plt.savefig('Amplitude.png', dpi=300)

fig, ax = plt.subplots()
ax.plot(xx, yyAnorm, 'g-')
ax.scatter(df.time.unique(), df[Anorm_name].unique(), label=Anorm_name)
ax.errorbar(df.time.unique(), df[Anorm_name].unique(), yerr=df['e_' + Anorm_name].unique(), ls='none')
plt.ylabel(Anorm_name)
plt.xlabel('time [us]')
title_str = 'Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    Anorm_popt[1], Anorm_perr[1], y_name, fit_func.__name__, boxsize)
plt.title(title_str)
plt.savefig('AmplitudeNorm.png', dpi=300)

print('f0_popt:')
print(f0_popt, f0_perr)

print('A_popt:')
print(A_popt, A_perr)

ps = A_popt[1] - f0_popt[1]
e_ps = np.sqrt(A_perr[1]**2 + f0_perr[1]**2)

title_str = 'phase shift: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    ps, e_ps, y_name, fit_func.__name__, boxsize)
print(title_str)

fig, ax1 = plt.subplots()

ax1.plot(xx, yyA, 'r-')
color = 'tab:red'
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylabel('Amplitude [arb.]', color=color)
ax1.set_xlabel('time [us]')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.plot(xx, yyf0, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Peak freq [MHz]', color=color)
ax2.set_title(title_str)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('phaseshift_comp.png', dpi=300)
plt.show()


def plot_gaussian(x, A, x0, sigma, C):
    return A * np.exp(-(x-x0)**2/(2*sigma**2)) + C


def plot_sinc2(x, A, x0, sigma, C):
    return A*(np.sinc((x-x0) / sigma)**2) + C


# stuff for poster
df1 = df[df['time'] == 20]
df2 = df[df['time'] == 70]
df3 = df[df['time'] == 120]

# calculate mean and std of points for plotting purposes
df1gb = df1.groupby('freq').agg({y_name:['mean','std','sem']}).reset_index()
df2gb = df2.groupby('freq').agg({y_name:['mean','std','sem']}).reset_index()
df3gb = df3.groupby('freq').agg({y_name:['mean','std','sem']}).reset_index()

fig, ax = plt.subplots()

xx = np.linspace(43, 43.4, 200)
if fit_func.__name__ == 'Gaussian':
    yy1 = plot_gaussian(
        xx, df1.A.values[0], df1.f0.values[0], df1.sigma.values[0], df1.C.values[0])
    yy2 = plot_gaussian(
        xx, df2.A.values[0], df2.f0.values[0], df2.sigma.values[0], df2.C.values[0])
    yy3 = plot_gaussian(
        xx, df3.A.values[0], df3.f0.values[0], df3.sigma.values[0], df3.C.values[0])
elif fit_func.__name__ == 'Sinc2':
    yy1 = plot_sinc2(
        xx, df1.A.values[0], df1.f0.values[0], df1.sigma.values[0], df1.C.values[0])
    yy2 = plot_sinc2(
        xx, df2.A.values[0], df2.f0.values[0], df2.sigma.values[0], df2.C.values[0])
    yy3 = plot_sinc2(
        xx, df3.A.values[0], df3.f0.values[0], df3.sigma.values[0], df3.C.values[0])
ax.errorbar(df1gb.freq, df1gb[y_name]['mean'], yerr = df1gb[y_name]['std'], marker='o', ls='none', \
            mfc = 'blue', mec ='black', mew=2, ecolor='blue', capsize=2)
ax.plot(xx, yy1, 'b-', label='t = 20 us')

ax.errorbar(df2gb.freq, df2gb[y_name]['mean'], yerr = df2gb[y_name]['std'], marker='o', ls='none', \
            mfc = 'green', mec ='black', mew=2, ecolor='green', capsize=2)
ax.plot(xx, yy2, 'g-', label='t = 70 us')

ax.errorbar(df3gb.freq, df3gb[y_name]['mean'], yerr = df3gb[y_name]['std'], marker='o', ls='none', \
            mfc = 'red', mec ='black', mew=2, ecolor='red', capsize=2)
ax.plot(xx, yy3, 'r-', label='t = 120 us')

ax.set_xlim([43.1, 43.3])
ax.set_xlabel('Freq [MHz]')
ax.set_ylabel('Atom num [arb.]')
plt.setp(ax.spines.values(), linewidth=2)
ax.legend()
plt.savefig('comp.png', dpi=300)
