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
from fit_functions import Gaussian, Sinc2, Dimerlineshape, Lorentzian
from library import FreqMHz
from data_class import Data
import matplotlib.colors as mc
import colorsys

# load data and set up
run = '2024-06-13_L'
boxsize = 'largebox'
run_fn = run + '_e_time=0.(\d+).dat'
meta_df = pd.read_excel('./contact_correlations/phaseshift/phaseshift_summary.xlsx')
meta_df = meta_df.loc[meta_df['filename'] == '2024-06-13_L_e.dat']
run_freq = meta_df.freq.values[0]
run_period = 1/run_freq * 1000 # us
x_name = "freq"
y_name = "c9"
# y_name = 'sum95'
fit_func = Gaussian
# fit_func = Dimerlineshape
# fit_func = Lorentzian
guess = [-5000, 43.2, 0.05, 25000]  # A, x0, sigma, C
data_folder = './contact_correlations/phaseshift/data/'
title_pre = run+'_' + y_name
# run = Data(run_fn, path=data_folder)
regex = re.compile(run_fn)

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
dB_perr_p = 0.03458996
# also checked that it matched independent cal within errorAoC
# dB_popt_fixed= np.array([ 5.60796011e-02,  2.10184021e+00, -6.29113593e-09])
# dB_perr_p = 0.07408817

def FixedSinkHzFixedPhase(t, A, C):
    omega = run_freq / 1000 * 2 * np.pi
    return A*np.sin(omega*t-2.2006124) + C

#array([ 0.1691759 ,  2.66058286, -0.30313304])
def FixedSinFreePhase(t, p):
    omega = run_freq / 1000 * 2 * np.pi
    A = 0.0498332
    C = -0.26532032
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

    subrun = Data(file, path=data_folder)
    # ms->us + 1/2length, won't be consistent across all data
    subrun.data.time = subrun.data.time * 1000 + \
        (meta_df.pulselength.values[0]/2.0) 
    subrun.data['timemodulo'] = subrun.data.time % run_period

    subrun.fit(fit_func, names=[x_name, y_name], guess=guess, label=str(
        subrun.data.time.iloc[0])+' us')

    [subrun.data['A'], subrun.data['f0'], subrun.data['sigma'], subrun.data['C']] = \
        [*subrun.popt]

    [subrun.data['e_A'], subrun.data['e_f0'], subrun.data['e_sigma'], subrun.data['e_C']] = \
        [*subrun.perr]
    
    # different ways of pseudonormalizing
    subrun.data['Aosum'] = subrun.data['A'] / subrun.data['sum95'].mean()
    subrun.data['AoROIsum'] = subrun.data['A'] / subrun.data['ROIsum'].mean()
    subrun.data['AoC'] = subrun.data['A'] / subrun.data['C']
    subrun.data['e_Aosum']= np.abs(subrun.data['Aosum'] * \
        np.sqrt((subrun.data['e_A']/subrun.data['A'])**2 + (subrun.data['sum95'].std()/subrun.data['sum95'].mean())**2  ))
    subrun.data['e_AoC']= np.abs(subrun.data['AoC'] * \
        np.sqrt((subrun.data['e_A']/subrun.data['A'])**2 + (subrun.data['e_C']/subrun.data['C'])**2  ))
    subrun.data['e_AoROIsum']= np.abs(subrun.data['AoROIsum'] * \
        np.sqrt((subrun.data['e_A']/subrun.data['A'])**2 + (subrun.data['ROIsum'].std()/subrun.data['ROIsum'].mean())**2))
    subrun_list.append(subrun.data)
    # plt.close('all')

df = pd.concat(subrun_list)

# skipset=np.array([0.030, 0.07, 0.011, 0.150]) * \
#       1000 + meta_df.pulselength.values[0]/2
# df = df[~df.time.isin(skipset)]

freqs = df.f0.unique()
amps = df.A.unique()
e_freqs = df.e_f0.unique()
e_amps = df.e_A.unique()

### SINUSOIDAL FITS TO RESULTS
# set modulo to an arbitrary high value larger than any time values to do normal time fit
# modulo = run_period 
modulo =10000
# times = df.time.unique() % modulo
times = df.time.unique()

# choose sin(wt-p) + C or A sin(wt) + B cos(wt) + C
sinefit_func = FixedSinkHz
# sinefit_func = FixedSinLinearSum
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
Anormfix_popt, Anormfix_pcov = curve_fit(FixedSinFreePhase, times % modulo, df[Anorm_name].unique(),sigma=df['e_'+Anorm_name].unique(), p0=[2.2])

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
    
    
#cheating
pAnormfix = Anormfix_popt[0]
pAnormfix_err = np.sqrt(np.diag(Anormfix_pcov))[0]

### PLOTTING RESULTS
xx = np.linspace(0, times.max(), 200)
yyf0 = sinefit_func(xx, *f0_popt)
yydB = sinefit_func(xx, *dB_popt)
yyA = sinefit_func(xx, *A_popt)
yyAnorm = sinefit_func(xx, *Anorm_popt)
yyAnormfix = FixedSinFreePhase(xx, *Anormfix_popt)

fig, axs = plt.subplots(3,2)
fig.suptitle('Signal: ' + y_name + ', Fit func: ' + fit_func.__name__)
axf = axs[0,0]
axb = axs[0,1]
axamp = axs[1,0]
axnamp = axs[1,1]
axcompamp = axs[2,0]
axcompnamp = axs[2,1]


# plot freq results
axf.plot(xx, yyf0, 'b-')
axf.scatter(times, freqs, label='f0')
axf.errorbar(times, freqs, yerr=e_freqs, ls='none')
axf.set_ylabel('f0 [MHz]')
axf.set_xlabel('time [us]')
# title_str = title_pre+' Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(pf0, pf0_err, y_name, fit_func.__name__, boxsize)
title_str = 'Phase: {:.2f} +/- {:.2f}'.format(pf0, pf0_err)
axf.set_title(title_str)
# plt.savefig(title_pre+'_Peak frequency.png', dpi=300)

# plot field results
axb.plot(xx, yydB, 'b-')
axb.scatter(times, df.dB.unique(), label='f0')
axb.errorbar(times, df.dB.unique(), yerr=df.e_dB.unique(), ls='none')
axb.set_ylabel('B-B0 [G]')
axb.set_xlabel('time [us]')
title_str = 'Phase: {:.2f} +/- {:.2f}'.format(pdB, pdB_err)
axb.set_title(title_str)
# plt.savefig(title_pre+'_magneticfield.png', dpi=300)

# plot amplitude results
# ax.plot(xx, yyf0, 'b--')
axamp.plot(xx, yyA, 'r-')
axamp.scatter(times, df.A.unique(), label='A')
axamp.errorbar(times, df.A.unique(), yerr=df.e_A.unique(), ls='none')
axamp.set_ylabel('A')
axamp.set_xlabel('time [us]')
title_str = 'Phase: {:.2f} +/- {:.2f}'.format(pA, pA_err)
axamp.set_title(title_str)
# plt.savefig(title_pre+'_Amplitude.png', dpi=300)

axnamp.plot(xx, yyAnorm, 'g-')
axnamp.scatter(times, df[Anorm_name].unique(), label=Anorm_name)
axnamp.errorbar(times, df[Anorm_name].unique(), yerr=df['e_' + Anorm_name].unique(), ls='none')
axnamp.set_ylabel(Anorm_name)
axnamp.set_xlabel('time [us]')
title_str = 'Phase: {:.2f} +/- {:.2f}'.format(pAnorm, pAnorm_err)
axnamp.set_title(title_str)
# plt.savefig(title_pre+'_AmplitudeNorm.png', dpi=300)
plt.tight_layout()

fig, ax = plt.subplots()
ax.plot(xx, yyAnormfix, 'c-')
ax.scatter(times, df[Anorm_name].unique(), label=Anorm_name)
ax.errorbar(times, df[Anorm_name].unique(), yerr=df['e_' + Anorm_name].unique(), ls='none')
plt.ylabel(Anorm_name)
plt.xlabel('time [us]')
title_str = title_pre+' free Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    pAnormfix, pAnormfix_err, y_name, fit_func.__name__, boxsize)
plt.title(title_str)

#plot offset results
fig, ax = plt.subplots()
ax.errorbar(times, df.C.unique(), yerr=df.e_C.unique(), marker='o', ls='none', \
            mfc = 'magenta', mec =adjust_lightness('magenta'), mew=2, ecolor='magenta', capsize=2 )
plt.ylabel('C')
plt.xlabel('time [us]')
plt.title(title_pre)
# plt.savefig(title_pre+'_offset.png',dpi=300)


### phase shifts
ps = pA - pdB
e_ps = np.sqrt(pA_err**2 + pdB_err**2)
ps_Anorm = pAnorm - pdB
e_psAnorm = np.sqrt(pAnorm_err**2 + pdB_err**2)
# ps_Anormfix = pAnormfix - dB_popt_fixed[1]
# e_psAnormfix = np.sqrt(pAnormfix_err**2 + dB_perr_p**2)


# phase shift from amplitude
#title_str = 'phase shift: {:.2f} +/- {:.2f}, {}, {}, {}'.format(ps, e_ps, y_name, fit_func.__name__, boxsize)
title_str = 'Phase shift: {:.2f} +/- {:.2f}'.format(ps, e_ps)
axcompamp.plot(xx, yydB, 'b-')
color = 'mediumblue'
axcompamp.tick_params(axis='y', labelcolor=color)
axcompamp.set_ylabel('B-B0 [G]', color=color)
axcompamp.set_xlabel('time [us]')
plt.setp(axcompamp.spines.values(), linewidth=2)
axcompamp2 = axcompamp.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
axcompamp2.plot(xx, yyA, color=color)
axcompamp2.tick_params(axis='y', labelcolor=color)
axcompamp2.set_ylabel('Amplitude [arb.]', color=color)
axcompamp2.set_title(title_str)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('phaseshift_comp.png', dpi=300)
# plt.show()

# # phase shift form normalized amplitude
#title_str = 'phase shift: {:.2f} +/- {:.2f}, {}, {}, {}'.format(ps_Anorm, e_psAnorm, y_name, fit_func.__name__, boxsize)
title_str = 'Phase shift: {:.2f} +/- {:.2f}'.format(ps_Anorm, e_psAnorm)
axcompnamp.plot(xx, yydB, 'b-')
color = 'mediumblue'
axcompnamp.tick_params(axis='y', labelcolor=color)
axcompnamp.set_ylabel('B-B0 [G]', color=color)
axcompnamp.set_xlabel('time [us]')
plt.setp(axcompnamp.spines.values(), linewidth=2)
axcompnamp2 = axcompnamp.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
axcompnamp2.plot(xx, yyAnorm, color=color)
axcompnamp2.tick_params(axis='y', labelcolor=color)
axcompnamp2.set_ylabel('Normalized Amplitude [arb.]', color=color)
axcompnamp2.set_title(title_str)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('phaseshift_Anorm_comp.png', dpi=300)
# plt.show()

plt.rcParams.update({'font.size': 14})


print(f'Run: {run}, freq: {run_freq}, A_popt: {A_popt}, PS: {ps}+/-{e_ps}')