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
import matplotlib.colors as mc
import colorsys
# module_folder = 'E:\\Analysis Scripts\\analysis'
# module_folder = '//Users//kevinxie//Documents//GitHub//analysis//phaseshift'
# if module_folder not in sys.path:
#     sys.path.insert(0, module_folder)

# load data and set up
run = '2024-04-30_K'
boxsize = 'midbox'
run_fn = run + '_e_' + boxsize+'_time=0.(\d+).dat'
meta_df = pd.read_excel('./contact_correlations/phaseshift/phaseshift_summary.xlsx')
meta_df = meta_df.loc[meta_df['filename'] == '2024-04-30_K_e.dat']
run_freq = meta_df.freq.values[0]
x_name = "freq"
y_name = "c9"
# y_name = 'sum95'
fit_func = Gaussian
guess = [-5000, 43.2, 0.02, 30000]  # A, x0, sigma, C
data_folder = './contact_correlations/phaseshift/data/'
title_pre = run+ '_' + y_name

regex = re.compile(run_fn)


### functions ###
# Fixed sinusoidal function depending on given wiggle freq
def FixedSinkHz(t, A, p, C):
    omega = run_freq / 1000 * 2 * np.pi  # kHz
    return A*np.sin(omega * t - p) + C


def BfromFmEB(f, b=9.51554, m=0.261):
    return (f + b) / m

def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
dB_popt_fixed = np.array([ 7.16815199e-02,  2.11861446e+00, -1.01003763e-12]) # from fitting the entire dataset
dB_perr_p = 0.038827
def FixedSinkHzFixedPhase(t, A, C):
    omega = run_freq / 1000 * 2 * np.pi
    return A*np.sin(omega*t-2.2006124) + C
subrun_list = []
for file in os.listdir(data_folder):
    res= regex.match(file)
    if not res: continue

    subrun = Data(file, path=data_folder)
    # ms->us + 1/2length, won't be consistent across all data
    subrun.data.time = subrun.data.time * 1000 + \
        (meta_df.pulselength.values[0]/2.0) + 200

    subrun.fit(fit_func, names=[x_name, y_name], guess=guess, label=str(
        subrun.data.time.iloc[0])+' us')

    [subrun.data['A'], subrun.data['f0'], subrun.data['sigma'], subrun.data['C']] = \
        [*subrun.popt]

    [subrun.data['e_A'], subrun.data['e_f0'], subrun.data['e_sigma'], subrun.data['e_C']] = \
        [*subrun.perr]
    
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

df = pd.concat(subrun_list)


skip_time = np.array([0.51,0.56,0.61]) * \
    1000 + meta_df.pulselength.values[0]/2
df = df[~df.time.isin(skip_time)]
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

# frequency sine fits
bounds = ([0, 0, -np.inf], [np.inf, 2*np.pi, np.inf])  # amp, phase, offset
f0_guess = [0.1, 1.5, 43.20]
f0_popt, f0_pcov = curve_fit(FixedSinkHz, df.time.unique(
), df.f0.unique(), sigma=df.e_f0.unique(), bounds=bounds, p0=f0_guess)

# field sine fits
bounds = ([0, 2.1, -np.inf], [np.inf, 2.3, np.inf])  # amp, phase, offset
B0 = BfromFmEB(f0_popt[2])
df['dB'] = BfromFmEB(df.f0) - B0
df['e_dB'] = BfromFmEB(df.f0 + df.e_f0) - BfromFmEB(df.f0)
dB_guess = [0.07, 2.1, 0]
dB_popt, dB_pcov = curve_fit(FixedSinkHz, df.time.unique(
), df.dB.unique(), sigma=df.e_dB.unique(), bounds=bounds, p0=dB_guess)
# amplitude sine fits
bounds = ([0, 0, -np.inf], [np.inf, 2*np.pi, np.inf])  # amp, phase, offset
A_guess = [1000, 2, -8000]
A_popt, A_pcov = curve_fit(FixedSinkHz, df.time.unique(
), df.A.unique(), sigma=df.e_A.unique(), bounds=bounds, p0=A_guess)
# amplitude norm sine fits
Anorm_guess = [0.2, 2, -0.5]
Anorm_name = 'AoC'
Anorm_popt, Anorm_pcov = curve_fit(FixedSinkHz, df.time.unique(
), df[Anorm_name].unique(), bounds=bounds, p0=Anorm_guess, sigma=df['e_'+Anorm_name].unique())

f0_perr = np.sqrt(np.diag(f0_pcov))
dB_perr = np.sqrt(np.diag(dB_pcov))
A_perr = np.sqrt(np.diag(A_pcov))
Anorm_perr = np.sqrt(np.diag(Anorm_pcov))

xx = np.linspace(0, df.time.max(), 200)
yyf0 = FixedSinkHz(xx, *f0_popt)
yydB = FixedSinkHz(xx, *dB_popt)
yyA = FixedSinkHz(xx, *A_popt)
yyAnorm = FixedSinkHz(xx, *Anorm_popt)

# plot freq results
fig, ax = plt.subplots()
ax.plot(xx, yyf0, 'b-')
ax.scatter(df.time.unique(), df.f0.unique(), label='f0')
ax.errorbar(df.time.unique(), df.f0.unique(), yerr=df.e_f0.unique(), ls='none')
plt.ylabel('f0 [MHz]')
plt.xlabel('time [us]')
title_str =title_pre+ ' Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    f0_popt[1], f0_perr[1], y_name, fit_func.__name__, boxsize)
plt.title(title_str)
#plt.savefig(title_pre+'_Peak frequency.png', dpi=300)

# plot field results
fig, ax = plt.subplots()
ax.plot(xx, yydB, 'b-')
ax.scatter(df.time.unique(), df.dB.unique(), label='f0')
ax.errorbar(df.time.unique(), df.dB.unique(), yerr=df.e_dB.unique(), ls='none')
plt.ylabel('B-B0 [G]')
plt.xlabel('time [us]')
title_str = title_pre+' Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    dB_popt[1], dB_perr[1], y_name, fit_func.__name__, boxsize)
plt.title(title_str)
#plt.savefig(title_pre+'_magneticfield.png', dpi=300)

# plot amplitude results
fig, ax = plt.subplots()
# ax.plot(xx, yyf0, 'b--')
ax.plot(xx, yyA, 'r-')
ax.scatter(df.time.unique(), df.A.unique(), label='A')
ax.errorbar(df.time.unique(), df.A.unique(), yerr=df.e_A.unique(), ls='none')
plt.ylabel('A')
plt.xlabel('time [us]')
title_str = title_pre+' Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    A_popt[1], A_perr[1], y_name, fit_func.__name__, boxsize)
plt.title(title_str)
#plt.savefig(title_pre+'Amplitude.png', dpi=300)

fig, ax = plt.subplots()
ax.plot(xx, yyAnorm, 'g-')
ax.scatter(df.time.unique(), df[Anorm_name].unique(), label=Anorm_name)
ax.errorbar(df.time.unique(), df[Anorm_name].unique(), yerr=df['e_' + Anorm_name].unique(), ls='none')
plt.ylabel(Anorm_name)
plt.xlabel('time [us]')
title_str = title_pre+' Phase: {:.2f} +/- {:.2f}, {}, {}, {}'.format(
    Anorm_popt[1], Anorm_perr[1], y_name, fit_func.__name__, boxsize)
plt.title(title_str)
#plt.savefig(title_pre+'_AmplitudeNorm.png', dpi=300)

#plot offset results
fig, ax = plt.subplots()
ax.errorbar(df.time.unique(), df.C.unique(), yerr=df.e_C.unique(), marker='o', ls='none', \
            mfc = 'magenta', mec =adjust_lightness('magenta'), mew=2, ecolor='magenta', capsize=2 )
plt.ylabel('C')
plt.xlabel('time [us]')
plt.title(title_pre)
#plt.savefig(title_pre+'_offset.png',dpi=300)

# ps = A_popt[1] - f0_popt[1]
# e_ps = np.sqrt(A_perr[1]**2 + f0_perr[1]**2)
# ps_Anorm = Anorm_popt[1] - f0_popt[1]
# e_psAnorm = np.sqrt(Anorm_perr[1]**2 + f0_perr[1]**2)
ps = A_popt[1] - dB_perr_p
e_ps = np.sqrt(A_perr[1]**2 + dB_perr_p**2)
ps_Anorm = Anorm_popt[1] - dB_popt_fixed[1]
e_psAnorm = np.sqrt(Anorm_perr[1]**2 + dB_perr_p**2)


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
ax2.plot(xx, yyA, color=color, ls='-')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Amplitude [arb.]', color=color)
ax2.set_title(title_str)
ax.tick_params(width=2)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.savefig('phaseshift_comp.png', dpi=300)
plt.show()

fig, ax1 = plt.subplots()
plt.rcParams.update({'font.size':14})
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
ax2.plot(xx, yyAnorm, color=color, ls='-')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('Amplitude [arb.]', color=color)
ax2.set_title(title_str)
ax1.tick_params(width=2)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('phaseshift_Anorm_comp.png', dpi=600)
#plt.savefig('phaseshift_Anorm_comp.pdf', format='pdf', bbox_inches="tight", dpi=600)
plt.show()


def plot_gaussian(x, A, x0, sigma, C):
    return A * np.exp(-(x-x0)**2/(2*sigma**2)) + C


def plot_sinc2(x, A, x0, sigma, C):
    return A*(np.sinc((x-x0) / sigma)**2) + C


# ## stuff for poster
# df1 = df[df['time'] == 220]
# df2 = df[df['time'] == 270]
# df3 = df[df['time'] == 320]

# # calculate mean and std of points for plotting purposes
# df1gb = df1.groupby('freq').agg({y_name:['mean','std','sem']}).reset_index()
# df1gb['dB'] = BfromFmEB(df1gb['freq']) - B0
# df2gb = df2.groupby('freq').agg({y_name:['mean','std','sem']}).reset_index()
# df2gb['dB'] = BfromFmEB(df2gb['freq']) - B0
# df3gb = df3.groupby('freq').agg({y_name:['mean','std','sem']}).reset_index()
# df3gb['dB'] = BfromFmEB(df3gb['freq']) - B0

# norm_num = df1gb[y_name,'mean'].max() 
# df1gb[y_name,'normmean'] = df1gb[y_name,'mean'] / norm_num
# df2gb[y_name,'normmean'] = df2gb[y_name,'mean'] / norm_num
# df3gb[y_name,'normmean'] = df3gb[y_name,'mean'] / norm_num

# fig, ax = plt.subplots()

# xx = np.linspace(-0.5, 0.5, 200)

# if fit_func.__name__ == 'Gaussian':
#     df1_popt,_ = curve_fit(plot_gaussian, df1gb.dB.values, df1gb[y_name, 'mean'].values, sigma= df1gb[y_name,'std'], p0 = [-1000, 0, 0.1, 12000])
#     df2_popt,_ = curve_fit(plot_gaussian, df2gb.dB.values, df2gb[y_name, 'mean'].values, sigma= df2gb[y_name,'std'], p0 = [-3000, 0, 0.01, 12000])
#     df3_popt,_ = curve_fit(plot_gaussian, df3gb.dB.values, df3gb[y_name, 'mean'].values, sigma= df3gb[y_name,'std'], p0 = [-1000, 0, 0.1, 12000])
#     yy1 = plot_gaussian(
#         xx, *df1_popt)
#     yy2 = plot_gaussian(
#         xx, *df2_popt)
#     yy3 = plot_gaussian(
#         xx, *df3_popt)
# elif fit_func.__name__ == 'Sinc2':
#     df1_popt,_ = curve_fit(plot_sinc2, df1gb.dB.values, df1gb[y_name, 'mean'].values, sigma= df1gb[y_name,'std'], p0 = [-1000, 0, 0.1, 12000])
#     df2_popt,_ = curve_fit(plot_sinc2, df2gb.dB.values, df2gb[y_name, 'mean'].values, sigma= df2gb[y_name,'std'], p0 = [-1000, 0, 0.1, 12000])
#     df3_popt,_ = curve_fit(plot_sinc2, df3gb.dB.values, df3gb[y_name, 'mean'].values, sigma= df3gb[y_name,'std'], p0 = [-1000, 0, 0.1, 12000])
#     yy1 = plot_sinc2(
#         xx, *df1_popt)
#     yy2 = plot_sinc2(
#         xx, *df2_popt)
#     yy3 = plot_sinc2(
#         xx, *df3_popt)
    
# # for plotting purposes
# arboff1 = 0
# arboff2 = 900
# arboff3 = -1500   

# color1='blue'
# color2='green'
# color3='red'

# ax.errorbar(df1gb.dB, df1gb[y_name]['mean'] + arboff1, yerr = df1gb[y_name]['std'], marker='o', markersize=6, ls='none', \
#             mfc = color1, mec =adjust_lightness(color1), mew=2, ecolor=color1, capsize=2)
# ax.plot(xx, yy1 + arboff1, color=color1, label='t = 20 us')

# ax.errorbar(df2gb.dB, df2gb[y_name]['mean'] + arboff2, yerr = df2gb[y_name]['std'], markersize=6, marker='o', ls='none', \
#             mfc = color2, mec =adjust_lightness(color2), mew=2, ecolor=color2, capsize=2)
# ax.plot(xx, yy2 + arboff2, color=color2, label='t = 70 us')

# ax.errorbar(df3gb.dB, df3gb[y_name]['mean'] + arboff3, yerr = df3gb[y_name]['std'], markersize=6, marker='o', ls='none', \
#             mfc = color3, mec =adjust_lightness(color3), mew=2, ecolor=color3, capsize=2)
# ax.plot(xx, yy3 + arboff3, color=color3, label='t = 120 us')

# ax.set_xlim([-0.4,0.4])
# ax.set_xlabel('B-B0 [G]')
# ax.set_ylabel('Atom num [arb.]')
# plt.setp(ax.spines.values(), linewidth=2)
# ax.tick_params(width=2)
# ax.legend()
# # plt.savefig('tucan_sample_spectra.png', dpi=500)
