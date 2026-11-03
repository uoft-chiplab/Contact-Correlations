#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Analyse wiggle phase shift measurements from Oct 27 2023, where binding energies and loss amplitude are fit for each time point in the wiggle.

"""
import sys
# module_folder = 'E:\\Analysis Scripts\\analysis'
module_folder = '//Users//kevinxie//Documents//GitHub//analysis//'
if module_folder not in sys.path:
    sys.path.insert(0, module_folder)
import re
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from fit_functions import Gaussian, Sinc2
from library import FreqMHz
from data_class import Data
import matplotlib.colors as mc
import colorsys


# load data and set up
run = '2024-04-25_D'
# boxsize = 'midbox'
run_fn = run + '_e.dat'
meta_df = pd.read_excel('phaseshift_summary.xlsx')
meta_df = meta_df.loc[meta_df['filename'] == run_fn]
run_freq = meta_df.freq.values[0]
run_period = 1/run_freq * 1000 # us
x_name = "freq"
name = "fraction95"
data_folder = 'data/'
title_pre = run+'_' + name

### functions ###
# Fixed sinusoidal function depending on given wiggle freq


def FixedSinkHz(t, A, p, C):
    omega = run_freq / 1000 * 2 * np.pi  # kHz
    return A*np.sin(omega * t - p) + C
    #return A * (np.sin(omega*t) * np.cos(p) - np.cos(omega*t)*np.sin(p)) + C


def BfromFmEB(f, b=9.51554, m=0.261):
    return (f + b) / m

def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

# dB_popt_fixed = np.array([ 7.12851462e-02,  2.20061239e+00, -3.89488857e-10]) # from fitting the entire dataset on May 2
dB_popt_fixed = np.array([0.035495, 1.97442, 202.152])
# also checked that it matched independent cal within error
dB_perr_p = 0.0524614
def FixedSinkHzFixedPhase(t, A, C):
    omega = run_freq / 1000 * 2 * np.pi
    return A*np.sin(omega*t-2.2006124) + C

#array([ 0.1691759 ,  2.66058286, -0.30313304])
def FixedSinFreePhase(t, p):
    omega = run_freq / 1000 * 2 * np.pi
    A = 0.00476365
    C = 0.92384509
    return A*np.sin(omega * t  - p) + C

def FixedSinLinearSum(t, A, c, s, C):
    '''
    A sin(wt - p) + C= A (cos(p)sin(wt) - sin(p)cos(wt)) + C
                     = A sin(wt) - B cos(wt) + C
    c: in-phase coefficient
    s: out-of-phase coefficient
    s/c = tan(p)
    '''
    omega = run_freq / 1000 * 2 * np.pi
    return A * (c*np.sin(omega*t) - s * np.cos(omega*t)) +C  

def chi2(times, obs, model, model_params):
    '''
    \sum_i (observed-expected)^2 / expected
    '''
    chi2 = 0
    for i, t in enumerate(times):
        exp = model(t, *model_params)
        chi2 += (obs[i] - exp)**2 / exp
        # print(exp)
        # print(obs[i])
        # print(chi2)
        
    return chi2


data = Data(run_fn, path=data_folder)
data.data['time'] = data.data['time'] *1000 + meta_df.pulselength.values[0]/2
time_cutoff = 1500
data.data = data.data[data.data['time'] < time_cutoff]
y_name = 'inv_' + name
data.data[y_name] = np.ones(len(data.data[name])) - data.data[name]
# y_name = name
df = data.data.groupby('time').agg({y_name:['mean','std','sem']}).reset_index()
times = df.time.values
# modulo = run_period
# times = times % modulo
means = df[y_name,'mean'].values
stds = df[y_name,'std'].values
sems = df[y_name,'sem'].values

sinefunc = FixedSinFreePhase
# sinefunc = FixedSinkHz
# for regular fitting
if sinefunc.__name__ == 'FixedSinkHz':
    bounds = ([0, 0, -np.inf], [np.inf, 2*np.pi, np.inf])  # amp, phase, offset
    f0_guess = [0.1, 1.5, 43.20]
    dB_guess = [0.07, 2.1, 0]
    A_guess = [1000, 2, -8000]
   
    popt, pcov = curve_fit(sinefunc, times, means, sigma=sems, bounds=bounds)
    phase = popt[1]
    e_phase= np.sqrt(np.diag(pcov))[1]
    xx = np.linspace(0, times.max(), 300)
    yy = sinefunc(xx, *popt)

# only free phase fitting
if sinefunc.__name__ == 'FixedSinFreePhase': 
    bounds = ([0],[2*np.pi])
    popt, pcov = curve_fit(sinefunc, times, means, sigma=sems, bounds=bounds)
    phase = popt[0]
    e_phase= np.sqrt(np.diag(pcov))[0]
    xx = np.linspace(0, times.max(), 300)

    yy = FixedSinFreePhase(xx, *popt)

    phis = np.linspace(1.0, 3.0, 20)
    chi2s = []
    color = 'tab:red'
    fig, ax = plt.subplots()
    ax.errorbar(times, means, yerr = sems, marker='o', markersize=6, ls='none', \
                mfc = color, mec =adjust_lightness(color), mew=2, ecolor=color, capsize=2)
    for i,phi in enumerate(phis):
        score = chi2(times, means, FixedSinFreePhase, [phi])
        chi2s.append(score)
        yy = FixedSinFreePhase(xx, phi)
        ax.plot(xx, yy, color = adjust_lightness(color, amount=i/4.0), label='$\phi={:.2f},\chi^2$ = {:.4f}'.format(phi,score))
    plt.xlabel('time [us]')
    plt.ylabel(y_name)
    ax.legend()
    
    fix, ax= plt.subplots()
    ax.scatter(phis, chi2s)
    idxmin = chi2s.index(min(chi2s))
    phase = phis[idxmin]
    yy = FixedSinFreePhase(xx, phase)

color = 'tab:red'
fig, ax = plt.subplots()

ax.errorbar(times,means, yerr = sems, marker='o', markersize=6, ls='none', \
            mfc = color, mec =adjust_lightness(color), mew=2, ecolor=color, capsize=2)
ax.plot(xx, yy, color=color, label='Fixed 2 kHz')
plt.ylabel(y_name)
plt.xlabel('time [us]')
title_str = title_pre+' Phase: {:.2f} +/- {:.2f}, {}'.format(
    phase, e_phase, y_name)
plt.title(title_str)
# plt.savefig(title_pre+'_Peak frequency.png', dpi=300)

yydB = FixedSinkHz(xx, *dB_popt_fixed)
yydB = yydB - yydB.mean() # kind of like B-B0

ps = phase - dB_popt_fixed[1]
e_ps = np.sqrt(e_phase**2 + dB_perr_p**2)

# phase shift comparison
fig, ax1 = plt.subplots()
plt.rcParams.update({'font.size': 14})
title_str = 'phase shift: {:.2f} +/- {:.2f}, {}'.format(
    ps, e_ps, y_name)
ax1.plot(xx, yydB, 'b-')
color = 'mediumblue'
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylabel('B-B0 [G]', color=color)
ax1.set_xlabel('time [us]')
plt.setp(ax1.spines.values(), linewidth=2)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.plot(xx, yy, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel(y_name + ' [arb.]', color=color)
ax2.set_title(title_str)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('phaseshift_Anorm_comp.png', dpi=300)
plt.show()
