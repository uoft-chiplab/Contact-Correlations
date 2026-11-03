#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Analyse wiggle phase shift measurements from Oct 27 2023, where binding energies and loss amplitude are fit for each time point in the wiggle.

"""
import sys
# module_folder = 'E:\\Analysis Scripts\\analysis'
# module_folder = '//Users//kevinxie//Documents//GitHub//analysis//'
# if module_folder not in sys.path:
#     sys.path.insert(0, module_folder)
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
run = '2024-05-02_B'
boxsize = 'midbox'
run_fn = run + '_e_' + boxsize+'_time=0.(\d+).dat'
meta_df = pd.read_excel('./contact_correlations/phaseshift/phaseshift_summary.xlsx')
meta_df = meta_df.loc[meta_df['filename'] == '2024-04-30_K_e.dat']
run_freq = meta_df.freq.values[0]
run_period = 1/run_freq * 1000 # us
x_name = "freq"
y_name = "c9"
# y_name = 'sum95'
fit_func = Gaussian
guess = [-5000, 43.2, 0.02, 30000]  # A, x0, sigma, C
data_folder = './contact_correlations/phaseshift/data/'
title_pre = run+'_' + y_name
regex = re.compile('2024-05-02_B_e_sig1_time=0.(\d+).dat')

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

dB_popt_fixed = np.array([ 7.12851462e-02,  2.20061239e+00, -3.89488857e-10]) # from fitting the entire dataset on May 2
# also checked that it matched independent cal within error
dB_perr_p = 0.03458996
def FixedSinkHzFixedPhase(t, A, C):
    omega = run_freq / 1000 * 2 * np.pi
    return A*np.sin(omega*t-2.2006124) + C

#array([ 0.1691759 ,  2.66058286, -0.30313304])
def FixedSinFreePhase(t, p):
    omega = run_freq / 1000 * 2 * np.pi
    A = 0.1691759
    C = -0.30313304
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

subrun_list = []
for file in os.listdir(data_folder):
    # print(file)
    res = regex.match(file)
    if not res:
        continue

    # fit each loss spectra with a naive Gaussian
    subrun = Data(file, path=data_folder)
    # ms->us + 1/2length, won't be consistent across all data
    subrun.data.time = subrun.data.time * 1000 + \
        (meta_df.pulselength.values[0]/2.0) 

    subrun.fit(fit_func, names=[x_name, y_name], guess=guess, label=str(
        subrun.data.time.iloc[0])+' us')

    [subrun.data['A'], subrun.data['f0'], subrun.data['sigma'], subrun.data['C']] = \
        [*subrun.popt]

    [subrun.data['e_A'], subrun.data['e_f0'], subrun.data['e_sigma'], subrun.data['e_C']] = \
        [*subrun.perr]
    
    # pseudo-normalize the amplitude by the offset C
    # subrun.data['Aosum'] = subrun.data['A'] / subrun.data['sum95'].mean()
    # subrun.data['AoROIsum'] = subrun.data['A'] / subrun.data['ROIsum'].mean()
    subrun.data['AoC'] = subrun.data['A'] / subrun.data['C']
    # subrun.data['e_Aosum']= np.abs(subrun.data['Aosum'] * \
        # np.sqrt((subrun.data['e_A']/subrun.data['A'])**2 + (subrun.data['sum95'].std()/subrun.data['sum95'].mean())**2  ))
    subrun.data['e_AoC']= np.abs(subrun.data['AoC'] * \
        np.sqrt((subrun.data['e_A']/subrun.data['A'])**2 + (subrun.data['e_C']/subrun.data['C'])**2  ))
    # subrun.data['e_AoROIsum']= np.abs(subrun.data['AoROIsum'] * \
        # np.sqrt((subrun.data['e_A']/subrun.data['A'])**2 + (subrun.data['ROIsum'].std()/subrun.data['ROIsum'].mean())**2))
    subrun_list.append(subrun.data)
    plt.close('all')

df = pd.concat(subrun_list)

### SKIPPING
# the atom number is only stable for certain sub-datasets, so we have to separate them
skipset1=np.array([0.01,0.06,0.11,0.16,0.21,0.26,0.31,0.36,0.41,0.46]) * \
     1000 + meta_df.pulselength.values[0]/2
skipset2 = np.array([0.08,0.18,0.28,0.38,0.48]) * \
    1000 + meta_df.pulselength.values[0]/2
skipset = skipset2
df = df[~df.time.isin(skipset)]

freqs = df.f0.unique()
amps = df.A.unique()
e_freqs = df.e_f0.unique()
e_amps = df.e_A.unique()

### SINUSOIDAL FITS TO RESULTS
# set modulo to an arbitrary high value larger than any time values to do normal time fit
modulo = run_period 
#modulo = 10000
# times = df.time.unique() % modulo
times = df.time.unique()

# choose sin(wt-p) + C or A sin(wt) + B cos(wt) + C
sinefit_func = FixedSinkHz
# sinefit_func = FixedSinLinearSum # this was useless but I'm keeping it for posterity
if sinefit_func.__name__ == 'FixedSinkHz': 
    bounds = ([0, 0, -np.inf], [np.inf, 2*np.pi, np.inf])  # amp, phase, offset
    f0_guess = [0.1, 1.5, 43.20]
    dB_guess = [0.07, 2.1, 0]
    A_guess = [1000, 2, -8000]
    Anorm_guess = [0.2, 2, -0.5]
    Anorm_name = 'AoC'
elif sinefit_func.__name__ == 'FixedSinLinearSum':
    bounds = ([0, -1, -1, -np.inf],[np.inf, 1, 1, np.inf])
    f0_guess = [0.1, 1, 1, 43.20]
    dB_guess = [ 0.07, 1, 1, 0]
    A_guess = [1000,1, 1, -5000]
    Anorm_guess = [0.2, 1, 1, -0.5]
    Anorm_name = 'AoC'

# frequency sine fits
f0_popt, f0_pcov = curve_fit(sinefit_func, times, freqs, sigma=e_freqs, bounds=bounds, p0=f0_guess)
# field sine fits
B0 = BfromFmEB(f0_popt[-1])
df['dB'] = BfromFmEB(df.f0) - B0
df['e_dB'] = BfromFmEB(df.f0 + df.e_f0) - BfromFmEB(df.f0)
dB_popt, dB_pcov = curve_fit(sinefit_func, times % modulo, df.dB.unique(), sigma=df.e_dB.unique(), bounds=bounds, p0=dB_guess)
# amplitude sine fits
A_popt, A_pcov = curve_fit(sinefit_func, times % modulo, amps, sigma=e_amps, bounds=bounds, p0=A_guess)
# amplitude norm sine fits
Anorm_popt, Anorm_pcov = curve_fit(sinefit_func, times % modulo, df[Anorm_name].unique(), bounds=bounds, p0=Anorm_guess, sigma=df['e_'+Anorm_name].unique())

# get errors
f0_perr = np.sqrt(np.diag(f0_pcov))
dB_perr = np.sqrt(np.diag(dB_pcov))
A_perr = np.sqrt(np.diag(A_pcov))
Anorm_perr = np.sqrt(np.diag(Anorm_pcov))

if sinefit_func.__name__ == 'FixedSinkHz':
    pf0 = f0_popt[1]
    pf0_err = f0_perr[1]
    pdB = dB_popt[1]
    pdB_err = dB_perr[1]
    pA = A_popt[1]
    pA_err = A_perr[1]
    pAnorm = Anorm_popt[1]
    pAnorm_err = Anorm_perr[1]
elif sinefit_func.__name__ == 'FixedSinLinearSum':
    pf0 = np.arctan(f0_popt[2]/f0_popt[1])
    pdB = np.arctan(dB_popt[2]/dB_popt[1])
    pA = np.arctan(A_popt[2]/A_popt[1])
    pAnorm = np.arctan(Anorm_popt[2]/Anorm_popt[1])
    pf0_err = 0
    pdB_err = 0
    pA_err = 0
    pAnorm_err = 0

### PLOTTING RESULTS
xx = np.linspace(0, times.max(), 200)
yyf0 = sinefit_func(xx, *f0_popt)
yydB = sinefit_func(xx, *dB_popt)
yyA = sinefit_func(xx, *A_popt)
yyAnorm = sinefit_func(xx, *Anorm_popt)

# plot freq results
fig, ax = plt.subplots()
ax.plot(xx, yyf0, 'b-')
ax.scatter(times, freqs, label='f0')
ax.errorbar(times, freqs, yerr=e_freqs, ls='none')
plt.ylabel('f0 [MHz]')
plt.xlabel('time [us]')
title_str = title_pre+' Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    pf0, pf0_err, y_name, fit_func.__name__, boxsize)
plt.title(title_str)
# plt.savefig(title_pre+'_Peak frequency.png', dpi=300)

# plot field results
fig, ax = plt.subplots()
ax.plot(xx, yydB, 'b-')
ax.scatter(times, df.dB.unique(), label='f0')
ax.errorbar(times, df.dB.unique(), yerr=df.e_dB.unique(), ls='none')
plt.ylabel('B-B0 [G]')
plt.xlabel('time [us]')
title_str = title_pre+' Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    pdB, pdB_err, y_name, fit_func.__name__, boxsize)
plt.title(title_str)
# plt.savefig(title_pre+'_magneticfield.png', dpi=300)

# plot amplitude results
fig, ax = plt.subplots()
# ax.plot(xx, yyf0, 'b--')
ax.plot(xx, yyA, 'r-')
ax.scatter(times, df.A.unique(), label='A')
ax.errorbar(times, df.A.unique(), yerr=df.e_A.unique(), ls='none')
plt.ylabel('A')
plt.xlabel('time [us]')
title_str = title_pre+' Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    pA, pA_err, y_name, fit_func.__name__, boxsize)
plt.title(title_str)
# plt.savefig(title_pre+'_Amplitude.png', dpi=300)

fig, ax = plt.subplots()
ax.plot(xx, yyAnorm, 'g-')
ax.scatter(times, df[Anorm_name].unique(), label=Anorm_name)
ax.errorbar(times, df[Anorm_name].unique(), yerr=df['e_' + Anorm_name].unique(), ls='none')
plt.ylabel(Anorm_name)
plt.xlabel('time [us]')
title_str = title_pre+' Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    pAnorm, pAnorm_err, y_name, fit_func.__name__, boxsize)
plt.title(title_str)
# plt.savefig(title_pre+'_AmplitudeNorm.png', dpi=300)

#plot offset results
fig, ax = plt.subplots()
ax.errorbar(times, df.C.unique(), yerr=df.e_C.unique(), marker='o', ls='none', \
            mfc = 'magenta', mec =adjust_lightness('magenta'), mew=2, ecolor='magenta', capsize=2 )
plt.ylabel('C')
plt.xlabel('time [us]')
plt.title(title_pre)
# plt.savefig(title_pre+'_offset.png',dpi=300)


### PHASE SHIFTS
ps = pA - dB_popt_fixed[1]
e_ps = np.sqrt(pA_err**2 + dB_perr_p**2)
ps_Anorm = pAnorm - dB_popt_fixed[1]
e_psAnorm = np.sqrt(pAnorm_err**2 + dB_perr_p**2)

# phase shift from amplitude
fig, ax1 = plt.subplots()
title_str = 'phase shift: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    ps, e_ps, y_name, fit_func.__name__, boxsize)
ax1.plot(xx, yydB, 'b-')
color = 'mediumblue'
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylabel('B-B0 [G]', color=color)
ax1.set_xlabel('time [us]')
plt.setp(ax1.spines.values(), linewidth=2)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.plot(xx, yyA, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Amplitude [arb.]', color=color)
ax2.set_title(title_str)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('phaseshift_comp.png', dpi=300)
plt.show()

# phase shift form normalized amplitude
fig, ax1 = plt.subplots()
plt.rcParams.update({'font.size': 14})
title_str = 'phase shift: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    ps_Anorm, e_psAnorm, y_name, fit_func.__name__, boxsize)
ax1.plot(xx, yydB, 'b-')
color = 'mediumblue'
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylabel('B-B0 [G]', color=color)
ax1.set_xlabel('time [us]')
plt.setp(ax1.spines.values(), linewidth=2)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.plot(xx, yyAnorm, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Normalized Amplitude [arb.]', color=color)
ax2.set_title(title_str)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# phase shift form normalized amplitude (phase as only free DOF version)
#cheating
# Anormfix_popt, Anormfix_pcov = curve_fit(FixedSinFreePhase, times % modulo, df[Anorm_name].unique(),sigma=df['e_'+Anorm_name].unique(), p0=[2.2])
# pAnormfix = Anormfix_popt[0]
# pAnormfix_err = np.sqrt(np.diag(Anormfix_pcov))[0]
# yyAnormfix = FixedSinFreePhase(xx, *Anormfix_popt)
# ps_Anormfix = pAnormfix - dB_popt_fixed[1]
# e_psAnormfix = np.sqrt(pAnormfix_err**2 + dB_perr_p**2)

# fig, ax = plt.subplots()
# ax.plot(xx, yyAnormfix, 'c-')
# ax.scatter(times, df[Anorm_name].unique(), label=Anorm_name)
# ax.errorbar(times, df[Anorm_name].unique(), yerr=df['e_' + Anorm_name].unique(), ls='none')
# plt.ylabel(Anorm_name)
# plt.xlabel('time [us]')
# title_str = title_pre+' free Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
#     pAnormfix, pAnormfix_err, y_name, fit_func.__name__, boxsize)
# plt.title(title_str)


# fig, ax1 = plt.subplots()
# plt.rcParams.update({'font.size': 14})
# title_str = 'phase shift: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
#     ps_Anormfix, e_psAnormfix, y_name, fit_func.__name__, boxsize)
# ax1.plot(xx, yydB, 'b-')
# color = 'mediumblue'
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_ylabel('B-B0 [G]', color=color)
# ax1.set_xlabel('time [us]')
# plt.setp(ax1.spines.values(), linewidth=2)
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:red'
# ax2.plot(xx, yyAnorm, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_ylabel('Normalized Amplitude [arb.]', color=color)
# ax2.set_title(title_str)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

print(f'Run: {run}, freq: {run_freq}, A_popt: {A_popt}, PS: {ps}+/-{e_ps}')