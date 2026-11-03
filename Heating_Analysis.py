# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Fits heating rates from unitary_expansion_analysis.m output .dat files.
Fit results are compiled and stored as a Dataframe in a pkl file for later 
analysis or plotting, and a xlsx for viewing.

"""
import os
import sys
# Analysis Scripts/
root = os.path.dirname(os.path.dirname(os.getcwd()))

# Fast-Modulation-Contact-Correlation-Project\analysis
analysis_folder = 'f:\\Analysis Scripts\\analysis'
if analysis_folder not in sys.path:
	sys.path.append(analysis_folder)

from data_class import Data
from library import deBroglie, deBroglie_kHz, a97, chi_sq, mK, hbar, h, kB, pi, \
	plt_settings, colors, markers, tintshade, tint_shade_color
from constants import *
from scipy.optimize import curve_fit
from itertools import chain
from matplotlib import cm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle

# # field wiggle calibration fit
# from contact_correlations.field_wiggle_calibration import Bamp_from_Vpp 

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
# put this repo first so contact_correlations.* resolves here, not to the old
# copy in f:\Analysis Scripts\analysis\contact_correlations
if root not in sys.path:
	sys.path.insert(0, root)

# select data folder
data_folder = 'contact_correlations\\heating'
# data_folder = 'data//reanalysed'
data_path = os.path.join(proj_path, data_folder)

### files and paths
pkl_filename = 'heating_rate_fit_results.pkl'
pkl_file = os.path.join(proj_path, pkl_filename)
xlsx_results_filename = 'heating_data_results.xlsx'
xlsx_results_file = os.path.join(proj_path, xlsx_results_filename)
metadata_filename = 'contact_correlations\\heating_metadata.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)

# Unitary Fermi Gas analysis code
from contact_correlations.UFG_analysis import BulkViscTrap, BulkViscUniform
UFG_pkl_filename = 'UFG.pkl'
UFG_pkl_file = os.path.join(proj_path, UFG_pkl_filename)

### load metadata dataframe
# metadata = pd.read_excel(metadata_file)

### script options
plotting = True
save = True
select_plotting = False
Load = False

##############################################
#### add list of files to analyze here #######
##############################################

# this selects all files that are not excluded e.g. have exclude = 0
files =  ['2026-10-30_G_e.csv','2026-10-30_I_e.csv','2026-10-30_K_e.csv']

### Functions
def calc_A(B0, T, e_T, Bamp, e_Bamp):
	''' Returns A and e_A '''
	A = deBroglie_kHz(T)/a97(B0-Bamp)
	e_a97 = np.abs(a97(B0-Bamp) - a97(B0-Bamp+e_Bamp)) # assumes symmetric
	e_dB = np.abs(deBroglie(T) - deBroglie(T+e_T)) # same
	e_A = A*np.sqrt((e_a97/a97(B0-Bamp))**2 + (e_dB/deBroglie(T))**2)
	return A, e_A

# shitty hard-coded linear fit of C/kFN vs. B field [G]
def calc_Ca(B0, Bamp):
	m = -2.07499885
	b = 420.18726779
	Ca = np.abs(m*Bamp)
	return Ca

# read description!!!
def fix_attribute(run, attr, attr_names):
	""" .... okay so like, if you're trying to generalize over multiscans, 
		you need to get the freq or amplitude from the .dat file, and use that 
		for analysis. That's what this does. It checks if those are not in the 
		metadata, and if they are't (i.e. they're nan) then it fills that attr
		with an entry from the .dat file, trying a list of names. """
	if getattr(run, attr) != getattr(run, attr): # nan check
		for name in attr_names:
			try:
				setattr(run, attr, run.data[name].values[0])
				break
			except KeyError:
				if name != attr_names[-1]:
					continue


def analysis_for_dof_equals_two(run):
	"""For use when dof = 2, i.e. you can't fit."""
	# dumbest shit I've ever written
	df1 = run.avg_data.loc[run.avg_data[run.param] == run.avg_data[run.param].min()]
	df2 = run.avg_data.loc[run.avg_data[run.param] == run.avg_data[run.param].max()]
	
	run.ToTF = (df2.ToTF.iloc[0]+df1.ToTF.iloc[0])/2
	run.e_ToTF = df1.em_ToTF.iloc[0]
	run.EF = (df2.EF.iloc[0]+df1.EF.iloc[0])/2/h/1e3
	run.e_EF = df1.em_EF.iloc[0]/h/1e3
	run.kF = np.sqrt(2*mK*h*1e3*run.EF)/hbar
	run.e_kF = (run.e_EF/run.EF)/2 * run.kF
	run.T = (df2['T'].iloc[0]+df1['T'].iloc[0])/2*kB/h/1e3 # .T is the transpose operation... LOL
	run.e_T = df1.em_T.iloc[0]*kB/h/1e3
	run.Ei = df1[settings['temp_param']].iloc[0]
	run.e_Ei = df1['em_'+settings['temp_param']].iloc[0]
	Ef = df2[settings['temp_param']].iloc[0]
	e_Ef = df2['em_'+settings['temp_param']].iloc[0]
	run.N = (df2.N.iloc[0]+df1.N.iloc[0])/2
	run.e_N = df1.em_N.iloc[0]
	Ni = df2.N.iloc[0] # this needs to be df1?
	e_Ni = df2.em_N.iloc[0]
	Nf = df2.N.iloc[0]
	e_Nf = df2.em_N.iloc[0]
	
	run.dof = 0
	run.chi_sq = 1
	
	run.Bamp, run.e_Bamp = Bamp_from_Vpp(run.Vpp, run.freq)
	run.lamda = np.sqrt(hbar/(mK*run.T*1e3))
	run.A, run.e_A = calc_A(run.B, run.T, run.e_T, run.Bamp, run.e_Bamp)
	run.Ca = calc_Ca(run.B, run.Bamp)
	# rise over run...
	deltax = (df2[run.param].iloc[0]-df1[run.param].iloc[0]) 
	deltay = Ef-run.Ei
	
	# I think this is done right... plz check.
	run.loss_rate = (Nf-Ni)/deltax/run.N
	run.e_loss_rate = run.loss_rate * np.sqrt((e_Nf/Nf)**2 + (run.e_N/run.N)**2)
	
	run.Edot = deltay/deltax # kHz^2
	run.e_Edot = run.Edot*np.sqrt((e_Ef/Ef)**2 + (run.e_Ei/run.Ei)**2)
	
	# Edot/EF**2/A**2
	run.rate = run.Edot/run.EF**2/run.A**2
	run.e_rate = run.rate*np.sqrt((run.e_Edot/run.Edot)**2+ \
							 (2*run.e_EF/run.EF)**2 + (2*run.e_A/run.A)**2)
	
def zeta_and_C(run, slope, freq):
	freq = freq #kHz
	# print(freq)
	EF = run['EF'][0]/1000/h #kHz 
	N = run['N'][0] #number
	T = run['T'][0] #Kevlin
	TkHz = T*kB/h/1e3 #kHz
	# print(TkHz)
	A, eA = calc_A(202.148, TkHz, 0, 0.024, 0.003) 
	kF = np.sqrt(2*mK*EF*h*1e3)/hbar
	lambda_T = deBroglie_kHz(TkHz)
	# print(lambda_T)
	# print(kF)
	Edot = 2 * N * EF**2 #i do not think this is defined correctly.... 
	heating = slope / EF**2 / A**2
	# print(slope)
	# asqzeta = (kF*lambda_T)**2/(9*pi*freq**2)*Edot*A**2
	asqzeta_rate = (kF*lambda_T)**2/(9*pi*freq**2)*slope/A**2
	# print(asqzeta)
	# print(asqzeta_rate)
	# zeta propto EF * Edot, so
	# e_zeta = asqzeta_rate*np.sqrt((e_EF/run.EF)**2+(e_Edot/Edot)**2)
	
	pi_factors = 2/9*36*pi*(2*pi)**(3/2)
	fandT_factors = (freq/TkHz)**(3/2)/(lambda_T*kF)/(2*pi*freq**2)
	# print((freq/TkHz)**(3/2))
	# C = pi_factors*fandT_factors*Edot/A**2 
	# print(lambda_T*kF)
	C_rate = pi_factors*fandT_factors*slope/A**2
	# print(C_rate)
	# C propto Edot/kF, so
	# e_C = C*np.sqrt((e_kF/kF)**2+(e_Edot/Edot)**2)
	return asqzeta_rate, C_rate, heating 

def linear(x,m,b):
	return m*x + b
	
############## FITTING ##############
plotting_params = ['ToTF', 'T', 'EFkHz','meanEtot_kHz','N', 'EkHz']
more_plotting = ['asqzeta','C','heating']
fitting_params = []
runs_data_dict = {}

colors_markers = ['midnightblue','darkmagenta','darkslategrey']
colors_lines = ['cornflowerblue','orchid','paleturquoise']
freqs = [10, 5, 20]
# loop over files, split into runs if they are multiscans
for i, file in enumerate(files):
    run = Data(file)
    # df_run = run.data
    print("Fitting data from {}".format(run.filename))
    
    xparam = 'time (ms)'
    # calculate correct energy from Will's code output in kHz
    run.data['meanEtot_kHz'] = run.data['ENoEF'] * run.data['EF']/1000/h
    
    # average data
    run.group_by_mean(xparam)
    df_run = run.avg_data

    for params in plotting_params:
        # print(params)

        # Perform the linear fit
        popt, pcov = curve_fit(linear, run.avg_data[xparam], run.avg_data[params], p0=[1, 0])
        perr = np.sqrt(np.diag(pcov))

        if params == 'EkHz':
            asqzeta, C, heating = zeta_and_C(df_run, popt[0], freqs[i])
            results = {            'run': file,
                       'asqzeta': asqzeta,
            'C': C,
            'heating': heating
            }
            fitting_params.append(results)

        result = {
			'run': file,
            'Variable': params,
            'slope': popt[0],
            'intercept': popt[1],
            'slope_err': perr[0],
            'intercept_err': perr[1],
        }
        fitting_params.append(result)
    df_results = pd.DataFrame(fitting_params)
    runs_data_dict[file] = df_run
    # print(df_results)

	### LOAD THEORY 
if Load == True:
    with open(UFG_pkl_file, 'rb') as f:  
        BVTs = pickle.load(f)
else:
    BVTs = []


############## PLOTTING ##############		
if plotting == True:
	### matplotlib options
    plt.rcParams.update({"figure.figsize": [12,6],
						  "legend.fontsize": 16,
						  "font.size": 16,
						   "errorbar.capsize": 0})

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True, sharex=True)
    fig, ax_zetaC = plt.subplots(2,2, figsize=(14,8))
    fig, ax_cycle = plt.subplots(1, figsize = (14,8))
	
    axs_flat = axs.flatten()
    axs_zetaC_flat = ax_zetaC.flatten()
    for j, runletter in enumerate(df_results[::len(plotting_params)+1]["run"]):
        run_temp = runs_data_dict[runletter]
        ToTF = run_temp['ToTF'][0]
        EF_run = run_temp['EF'][0]/1000/h #kHz
        if Load == False: # calculate trap theory
            barnu = (111*400*500)**(1/3)
            nu_min = 1
            nu_max = 140
            num = nu_max-nu_min+1
            nus = np.linspace(nu_min, nu_max, num)
            BVT = BulkViscTrap(ToTF, EF_run, barnu, nus)	
        else:
            for BVT in BVTs:
                if BVT.ToTF == ToTF:
                    foundBVT = True
                    print(BVT.ToTF)
                    break
                if not foundBVT: # if can't find, then throw error
                    raise FileNotFoundError("No BVT in pickle matches ToTF {ToTF} \n Recalculate BVTs.")
                
        for i, params in enumerate(plotting_params):
            df_results_temp = df_results[df_results['run'] == runletter]
            
            # print(run_temp)
            ax = axs_flat[i]
            
            xx = np.array(run_temp[xparam])
            yy = np.array(run_temp[params])

            
            row = df_results_temp[df_results_temp['Variable'] == params]
            m = row['slope'].item()
            b = row['intercept'].item()
            m_err = row['slope_err'].item()
            b_err = row['intercept_err'].item()
            popt = [m, b]
            perr = [m_err,b_err]
            xs = np.linspace(min(xx),max(xx),100)
            y_fits = linear(xs, *popt)
            # Plot the averaged data and fits
            ax.errorbar(xx, yy, yerr=m_err, marker='o', linestyle='', 
                        markerfacecolor=colors_markers[j],markeredgecolor = colors_markers[j],
                        color = colors_markers[j],
                    label=f'Freq = {freqs[j]}'
                    )
            ax.plot(xs, y_fits, marker='', linestyle='-', color = colors_lines[j]
                    # label=params
                    )        
            # ax.set_xlabel(xparam)
            ax.set_ylabel(params)
        for i, plot in enumerate(more_plotting):
            #plot theory for each ToTF of each run
            ylabels = [f'$a^2\zeta$','C','Heating $\dot{E}/EF^2/A^2$']
            axs_zetaC = axs_zetaC_flat[i]
            freq = freqs[j]
            # for nu in nus.tolist():
            #     heating_theory = [BVT.calc_Edot(nu)]
            xt = BVT.nus /BVT.EF
            yt =  BVT.zeta
            xx = freq/EF_run
            yy = np.mean(df_results_temp[plot])

            
            axs_zetaC.plot(xx, yy, color = colors_markers[j],
                                #    label=plot
                                   )
            axs_zetaC.set_ylabel(ylabels[i])
            
            if i == 0:
                axs_zetaC.set_yscale('log')
                axs_zetaC.plot(xt, yt, '--', color=colors_lines[j], 
                               label = f'ToTF = {ToTF:.3f}'
                               )
                axs_zetaC.legend()
                axs_zetaC.set_xscale('log')
            if i == 2:
                axs_zetaC.plot(xt, BVT.Edot, '--', color=colors_lines[j], 
                               label = f'ToTF = {ToTF:.3f}'
                               )
                axs_zetaC.set_xlim([0,6])
            
			
    ax.legend()
    
    for ax in axs[-1,:]:
        ax.set_xlabel(xparam)
    for ax in axs_zetaC_flat[1:]:
        ax.set_xlabel(f'$\omega/E_F$')
    # Hide any remaining unused subplots in the 2x3 grid
    # for j in range(len(plotting_params), len(axs_flat)):
    #     fig.delaxes(axs_flat[j])

		

    # fig.suptitle(files[0][0:12]+' 20kHz, 0.5Vpp')