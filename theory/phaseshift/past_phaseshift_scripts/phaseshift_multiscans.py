#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Analyse wiggle phase shift measurements from Oct 27 2023, where binding energies and loss amplitude are fit for each time point in the wiggle.

"""

from data_class import Data
from library import FreqMHz
from analysisfunctions import FixedSin5kHz

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

### functions ###
def FixedSin10kHz():
	

def wigglefield_from_time(t, arb_scaling=1):
    omega=2*np.pi*5000
    amp=0.0695243 #+/- 0.0025798
    phase=2.17306 #+/- 0.0474719
    offset=202.072 #+/- 0.00203875
    return arb_scaling * amp * np.sin(omega * (t*10**-6) - phase) + offset

def Eb_from_field(B, whichfit='linear'):
    '''
    Eb cal from Oct 30
    Eb in MHz, field input in G
    '''
    if whichfit == 'linear':
        a=-0.05124 #+/- 0.006695
        b=14.35 #+/- 1.353
        return a*B + b
    
def Gaussian(x, A, x0, sigma, C):
	"""
	Returns:  A * np.exp(-(x-x0)**2/(2*sigma**2)) + C
	"""
	return A * np.exp(-(x-x0)**2/(2*sigma**2)) + C

plt.style.use('plottingstype.mplstyle')

### switches ###
FIX_EB=False
FIX_WIDTH=False
fixedwidth=0.034542573613013806 #taken from the median sigma of an unfixed result
save_intermediate=False
save_final=False
if FIX_EB and FIX_WIDTH:
    suffix_str = 'f0 fixed and sigma fixed'
elif FIX_EB:
    suffix_str = 'f0 fixed'
elif FIX_WIDTH:
    suffix_str = 'sigma fixed'
else:
    suffix_str = 'unfixed'
    
### data setup ###
data_folder = 'data/oct27phaseshift'
df = pd.read_csv(data_folder + '/Oct27summary.csv')

df.drop([0,3,15,16],inplace=True) # B, C, F, E were bad runs
df.sort_values(by='delay + pulse', inplace=True)
df.reset_index(drop=True, inplace=True)

df['field']=wigglefield_from_time(df['delay + pulse'])
df['Eb']=Eb_from_field(df['field'])
df['f75'] = FreqMHz(df['field'], 9/2, -5/2, 9/2, -7/2)
df['f75-Eb']=df['f75']-df['Eb']

#print(df)

# initialize lists
meass=['c5','c9','sum95']
popt_list=[]
perr_list=[]
run_list=[]
fit_dict_list=[]

### begin analysis ###
for irow in range(0, len(df)):
    run_letter=df.iloc[irow]['run']
    run_time = df.iloc[irow]['delay + pulse']
    run_filename='2023-10-27_' + run_letter + '_e.dat'
    run=Data(run_filename, path=data_folder, column_names=['freq (MHz)', *meass])
    run.field=df.loc[irow, 'field']
    run.peakfreq=df.loc[irow, 'f75-Eb']

    for i,meas in enumerate(meass):
    
        # adjusts the fit function depending on fixed fit parameter
        if FIX_EB and FIX_WIDTH:
            popt, pcov = curve_fit(lambda f, amp, offset: Gaussian(f, amp, run.peakfreq, fixedwidth, offset), \
                                   run.data['freq (MHz)'], run.data[meas],p0=[-3000, 0])
            perr = np.sqrt(np.diag(pcov))
            fit_dict = {'run': run_letter, 'meas': meas, 'amp': popt[0], 'offset': popt[1], \
                        'e_amp':perr[0], 'e_offset' : perr[1]}    
            show_popt = np.insert(popt,1, run.peakfreq)
            show_popt = np.insert(show_popt, 2, fixedwidth)
        elif FIX_EB == True:
            popt, pcov = curve_fit(lambda f, amp, sigma, offset: Gaussian(f, amp, run.peakfreq, sigma, offset), \
                                   run.data['freq (MHz)'], run.data[meas],p0=[-3000, fixedwidth, 0])
            perr = np.sqrt(np.diag(pcov))
            fit_dict = {'run': run_letter, 'meas': meas, 'amp': popt[0], 'sigma' : popt[1], 'offset': popt[2], \
                        'e_amp':perr[0], 'e_sigma' : perr[1], 'e_offset' : perr[2]}    
            show_popt = np.insert(popt,1, run.peakfreq)
            
        elif FIX_WIDTH == True:
            popt, pcov = curve_fit(lambda f, amp, f0, offset: Gaussian(f, amp, f0, fixedwidth, offset), \
                                   run.data['freq (MHz)'], run.data[meas],p0=[-3000, run.peakfreq, 0])
            perr = np.sqrt(np.diag(pcov))
            fit_dict = {'run': run_letter, 'meas': meas, 'amp': popt[0], 'f0':popt[1], 'offset': popt[2], \
                     'e_amp':perr[0], 'e_f0' : perr[1],'e_offset' : perr[2]}
            show_popt = np.insert(popt,2, fixedwidth)
        else:
            popt, pcov = curve_fit(Gaussian, run.data['freq (MHz)'], run.data[meas],p0=[-3000, run.peakfreq, 0.05, 0])
            perr = np.sqrt(np.diag(pcov))
            fit_dict = {'run': run_letter, 'meas': meas, 'amp': popt[0], 'f0':popt[1], 'sigma' : popt[2], 'offset': popt[3], \
                        'e_amp':perr[0], 'e_f0' : perr[1], 'e_sigma' : perr[2], 'e_offset' : perr[3]}
            show_popt = popt
        
        fit_dict_list.append(fit_dict)
    
        if save_intermediate:
            run_title = run_letter + '_' + str(run_time) + '_' + meas + '_'+ suffix_str
            fpoints = np.linspace(run.data['freq (MHz)'].min(), run.data['freq (MHz)'].max(),200)
            fig, ax= plt.subplots(1,1)
            ax.plot(fpoints, Gaussian(fpoints, *show_popt), 'r-')
            ax.scatter(run.data['freq (MHz)'], run.data[meas],label=run_filename)
            ax.set_ylabel(meas)
            ax.set_xlabel('freq (MHz)')
            ax.legend(loc='upper right')
            ax.set_title(run_title)
            plt.savefig(data_folder + '/results/' +run_title + '.png')
        
# merge fit results into original dataframe
# should have 3 rows for each run, 1 for each measurable
fit_df = pd.DataFrame(fit_dict_list)
df = pd.merge(df, fit_df, on='run',how='outer')
       
### Sinusoidal fit to fit results over time ###
if FIX_EB and FIX_WIDTH:
    params=['amp','offset']
elif FIX_EB:
    params=['amp','sigma','offset']
elif FIX_WIDTH:
    params=['amp','f0','offset']
else:
    params=['amp','f0','sigma','offset']
    
dflist=[]
for param in params:
    bounds=([-np.inf, 0, -np.inf],[np.inf, np.pi, np.inf]) # ensure phase > 0

    c5_df = df[df.meas=='c5']
    c9_df = df[df.meas=='c9']
    sum95_df = df[df.meas == 'sum95']
    
    c5_popt, c5_pcov = curve_fit(FixedSin5kHz, c5_df['delay + pulse'], c5_df[param], sigma=c5_df['e_'+param],bounds=bounds, p0=[100, 2,43])
    c5_perr = np.sqrt(np.diag(c5_pcov))
    
    c9_popt, c9_pcov = curve_fit(FixedSin5kHz, c9_df['delay + pulse'], c9_df[param], sigma=c9_df['e_' + param],bounds=bounds,p0=[100, 2,43])
    c9_perr = np.sqrt(np.diag(c9_pcov))
    
    sum95_popt, sum95_pcov = curve_fit(FixedSin5kHz, sum95_df['delay + pulse'], sum95_df[param], sigma=sum95_df['e_' + param],bounds=bounds,p0=[100, 2,43])
    sum95_perr = np.sqrt(np.diag(sum95_pcov))
    
    temp_df = pd.DataFrame([{'meas':'c5', 'param':param, 'popt':c5_popt,'perr':c5_perr},\
                           {'meas':'c9',  'param':param,'popt':c9_popt,'perr':c9_perr},\
                               {'meas':'sum95',  'param':param,'popt':sum95_popt,'perr':sum95_perr}])  
    dflist.append(temp_df)
    
# summarize phase shift results
ps_df = pd.concat(dflist)
ps_df['phase']=ps_df.popt.map(lambda x: x[1])
ps_df['e_phase']=ps_df.perr.map(lambda x: x[1])

##### Plot fit results over time #####
title_str = 'Phase shift_' + suffix_str
### AMPLITUDE
param='amp'
arb_scaling=24000
arb_offset=-3500
tpoints = np.linspace(-100,300,400)
fig, ax=plt.subplots(1,1)
ax.plot(tpoints, wigglefield_from_time(tpoints, arb_scaling)+arb_offset, 'b-', label='field')
colors = ["red", 
		  "green", "orange", 
		  "purple", "black", "pink"]
linestyles=['r-','g-','y-','p-','k-','m-']
for i,meas in enumerate(meass):
    
    if meas == 'c5':
        legendtxt = 'b'
        plot_df = c5_df
    elif meas == 'c9':
        legendtxt = 'a'
        plot_df = c9_df
    else :
        legendtxt='a+b'
        plot_df = sum95_df
    ax.errorbar(plot_df['delay + pulse'], plot_df[param],plot_df['e_'+param],color=colors[i], label=legendtxt)
    ax.plot(tpoints, FixedSin5kHz(tpoints, *ps_df[(ps_df.meas==meas) & (ps_df.param == param)]['popt'].values[0]), linestyles[i])

ax.legend(loc='upper right')
ax.set_xlabel('time (us)')
ax.set_ylabel('amplitude (arb.)')
ax.set_title(title_str)
if save_final:
    plt.savefig(data_folder + '/results/' + title_str + '_amp.png')

### Plot peak frequency fit results ###
### FREQUENCY ###
if FIX_EB == False:
    param='f0'
    arb_scaling=0.3
    arb_offset=-158.86
    tpoints = np.linspace(-100,300,400)
    fig, ax=plt.subplots(1,1)
    ax.plot(tpoints, wigglefield_from_time(tpoints, arb_scaling)+arb_offset, 'b-', label='field')
    for i,meas in enumerate(meass):
        if meas == 'c5':
            legendtxt = 'b'
            plot_df = c5_df
        elif meas == 'c9':
            legendtxt = 'a'
            plot_df = c9_df
        else :
            legendtxt='a+b'
            plot_df = sum95_df
        ax.errorbar(plot_df['delay + pulse'], plot_df[param],plot_df['e_'+param],color=colors[i], label=legendtxt)
        ax.plot(tpoints, FixedSin5kHz(tpoints, *ps_df[(ps_df.meas==meas) & (ps_df.param == param)]['popt'].values[0]), linestyles[i])
    
    ax.legend(loc='upper right')
    ax.set_xlabel('time (us)')
    ax.set_ylabel('f0 (MHz)')
    ax.set_title(title_str)
    if save_final:
        plt.savefig(data_folder + '/results/' + title_str + '_f0.png')
    
### SIGMA
if FIX_WIDTH==False:
    param='sigma'
    arb_scaling=0.1
    arb_offset=-202.04
    tpoints = np.linspace(-100,300,400)
    fig, ax=plt.subplots(1,1)
    ax.plot(tpoints, wigglefield_from_time(tpoints, arb_scaling)+arb_offset, 'b-', label='field')
    colors = ["red", 
    		  "green", "orange", 
    		  "purple", "black", "pink"]
    linestyles=['r-','g-','y-','p-','k-','m-']
    for i,meas in enumerate(meass):
        
        if meas == 'c5':
            legendtxt = 'b'
            plot_df = c5_df
        elif meas == 'c9':
            legendtxt = 'a'
            plot_df = c9_df
        else :
            legendtxt='a+b'
            plot_df = sum95_df
        ax.errorbar(plot_df['delay + pulse'], plot_df[param],plot_df['e_'+param],color=colors[i], label=legendtxt)
        ax.plot(tpoints, FixedSin5kHz(tpoints, *ps_df[(ps_df.meas==meas) & (ps_df.param == param)]['popt'].values[0]), linestyles[i])
    
    ax.legend(loc='upper right')
    ax.set_xlabel('time (us)')
    ax.set_ylabel('sigma (MHz)')
    ax.set_ylim([0.02,0.07])
    ax.set_title(title_str)
    if save_final:
        plt.savefig(data_folder + '/results/' + title_str + '_sigma.png')


### CONSTANT
param='offset'
arb_scaling=80000
arb_offset=14000
tpoints = np.linspace(-100,300,400)
fig, ax=plt.subplots(1,1)
ax.plot(tpoints, wigglefield_from_time(tpoints, arb_scaling)+arb_offset, 'b-', label='field')
colors = ["red", 
		  "green", "orange", 
		  "purple", "black", "pink"]
linestyles=['r-','g-','y-','p-','k-','m-']
for i,meas in enumerate(meass):
    
    if meas == 'c5':
        legendtxt = 'b'
        plot_df = c5_df
    elif meas == 'c9':
        legendtxt = 'a'
        plot_df = c9_df
    else :
        legendtxt='a+b'
        plot_df = sum95_df
    ax.errorbar(plot_df['delay + pulse'], plot_df[param],plot_df['e_'+param],color=colors[i], label=legendtxt)
    ax.plot(tpoints, FixedSin5kHz(tpoints, *ps_df[(ps_df.meas==meas) & (ps_df.param == param)]['popt'].values[0]), linestyles[i])

ax.legend(loc='upper right')
ax.set_xlabel('time (us)')
ax.set_ylabel('offset (arb.)')
ax.set_title(title_str)
if save_final:
    plt.savefig(data_folder + '/results/' + title_str + '_offset.png')