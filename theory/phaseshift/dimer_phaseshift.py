# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:48:13 2024

@author: coldatoms

Refactored analysis script for dimer phase shift measurements of April-June 2024.
"""

# paths

import os
analysis_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '\\analysis'
import sys
if analysis_folder not in sys.path:
	sys.path.append(analysis_folder)
data_path = r'\\UNOBTAINIUM\E_Carmen_Santiago\Analysis Scripts\Fast-Modulation-Contact-Correlation-Project\contact_correlations\phaseshift\data'
from library import pi, h, hbar, mK, a0, plt_settings, styles, colors
#from fit_functions import Gaussian, Sinc2
from data_helper import remove_indices_formatter
from save_df_to_xlsx import save_df_row_to_xlsx
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from scipy.optimize import curve_fit
from warnings import catch_warnings, simplefilter
from cycler import Cycler, cycler
from math import ceil
from math import isnan
from inspect import signature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import gc
gc.collect()

	
### This turns on (True) and off (False) saving the data/plots 
SAVE_RESULTS= False
SAVE_INTERMEDIATE = False
### script options
DEBUG = False
FILTER = True
TALK = False
REEVALUATE = True
PLOT_SUMMARY = False
ana_str = 'phaseshift'
#lineshape = 'gaussian_fixedwidth'
lineshape = 'sinc2'
#lineshape='gaussian'
# select spin state imaging for analysis
state_selection = '97'

#spins = ['c5', 'c9', 'sum95']
#spins = ['c5', 'c9', 'ratio95']
spins = ['c5']

# freq_name = 'detuning_EF' #'freq'/detuning
freq_name = 'freq'
### metadata
metadata_filename = ana_str+'_metadata.xlsx'
proj_path = os.path.dirname(__file__) + '\\phaseshift'
metadata_file = os.path.join(proj_path, metadata_filename)
metadata = pd.read_excel(metadata_file)
files =  metadata.loc[metadata['exclude'] == 0]['filename'].values
files = ["2024-05-10_V_e"]

def spin_map(spin):
	if spin == 'c5':
		return 'b'
	elif spin == 'c9':
		return 'a'
	elif spin == 'sum95':
		return 'a+b'
	elif spin == 'ratio95' or spin == 'dimer':
		return 'a/b'
	else:
		return ''
	
# save file paths
savefilename = ana_str + '_analysis_results_' + state_selection + '.xlsx'
savefile = os.path.join(proj_path, savefilename)

pkl_file = os.path.join(proj_path, ana_str+'_results.pkl')

### Vpp calibration
# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
VpptoOmegaR47 = 17.05/0.703 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
VpptoOmegaR43 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration 
phaseO_OmegaR = lambda VVA, freq: 2*pi*VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)

CORRECT_SPINLOSS = True
SATURATION_CORRECTION = True

# Omega^2 [kHz^2] 1/e saturation value fit from 09-26_F
dimer_x0 = 4071
dimer_e_x0 = 1058
# Omega^2 [kHz^2] 1e saturation value fit from 09-17_C
HFT_x0 = 805.2923
HFT_e_x0 = 19.1718 

def saturation_scale(x, x0):
	""" x is OmegaR^2 and x0 is fit 1/e Omega_R^2 """
	return x/x0*1/(1-np.exp(-x/x0))
	
### constants
re = 107 * a0 # ac dimer range estimate
Eb = 3.98 # MHz # I guesstimated this from recent ac dimer spectra
kF = 1.1e7
kappa = np.sqrt((Eb*h*10**6) *mK/hbar**2) # convert Eb back to kappa
# actually the above constants should be rewrittento accout for different Bfields,
# becauese some of the data is taken at 202 G.

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))

def xstar(B, EF):
	return Eb/EF # hbar**2/mK/a13(B)**2 * (1-re/a13(Bfield))**(-1)

def linear(x, a, b):
	return a*x + b

def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer

def dimer_transfer(Rab, fa, fb):
	'''computes transfer to dimer state assuming loss in b is twice the loss
		in a. Rba = Nb/Na for the dimer association shot. Nb_bg and Na_bg
		are determined from averaged bg shots.'''
	return (fa - Rab*fb)/(1/2-Rab)

# this calibration is quite old and rough  -- from ??
def BfromFmEB(f, b=9.51554, m=0.261):
    return (f + b) / m

### plot settings
plt.rcParams.update(plt_settings) # from library.py

plt.rcParams.update({"figure.figsize": [12,8],
					 "font.size": 14,
					 "lines.markeredgewidth": 2,
					 "errorbar.capsize": 0})

### Summary plot lists
results_list = []

try:
	with open(pkl_file, 'rb') as f:
	    old_results_list = pkl.load(f)
except (OSError, IOError) as e:
    old_results_list = []

if REEVALUATE == True:
	old_results_list = []

### loop analysis over selected datasets
save_df_index = 0
for filename in files:
		
##############################
######### Analysis ###########
##############################
	print("----------------")
	print("Analyzing " + filename)
	
	meta_df = metadata.loc[metadata.filename == filename].reset_index()
	if meta_df.empty:
		print("Dataframe is empty! The metadata likely needs updating." )
		continue
	
	def FixedSinkHz(t, A, p, C):
		omega = meta_df['drive_freq'][0] / 1000 * 2 * np.pi  # kHz
		return A*np.sin(omega * t - p) + C
	
	if SAVE_INTERMEDIATE==True :
		interm_folder = os.path.join(data_path, filename)
		interm_pkl_file = os.path.join(interm_folder, filename+'.pkl')
		if not os.path.exists(interm_folder):	
			os.makedirs(os.path.join(data_path, filename))
			print('Made a subdirectory for this run in the data directory.')
	
	if filename[-4:] != ".dat":
		filename = filename + ".dat"
	
	# check if we need to evalute this run
	skip_evaluation = False
	for old_results in old_results_list:
		if old_results['Run'] == filename:
			print("Already evaluated, skipping...")
			results_list.append(old_results)
			skip_evaluation = True
			break
	if skip_evaluation:
		continue
	
	### if the current run's data files are segregated by time,
	# loop over them and combine them into one .dat file
	if not os.path.exists(os.path.join(data_path, filename)) and (meta_df['multifiles?'][0] == 1):
		import re
		subfile_re = filename[0:-4] + '(.*)time(.*).dat'
		regex = re.compile(subfile_re)
		sub_ls = []
		for subfile in os.listdir(data_path):
			re_match = regex.match(subfile)
			if not re_match:
				continue
			else:
				print(subfile)
				sub_df = pd.read_table(os.path.join(data_path, subfile), delimiter=',')
				sub_ls.append(sub_df)
		output_df = pd.concat(sub_ls)
		output_df.to_csv(f'{data_path}/{filename}', index=False)
		
	# create data structure
	run = Data(filename, path=data_path)
	print(filename)
	runfolder = filename 
	
	# initialize results dict to turn into df
	results = {
			'Run': filename,
			'pulse_type':meta_df['pulse_type'][0],
			'drive_freq':meta_df['drive_freq'][0],
			'trf': meta_df['trf_us'][0],
			'ToTF_i': meta_df['ToTF_i'][0],
			'ToTF_f': meta_df['ToTF_f'][0],
			'ToTF_diff': meta_df['ToTF_f'][0]-meta_df['ToTF_i'][0],
			'ToTF': (meta_df['ToTF_i'][0] + meta_df['ToTF_f'][0])/2,
			'e_ToTF':np.sqrt(meta_df['ToTF_i_sem'][0]**2 + \
							 meta_df['ToTF_f_sem'][0]**2)/2,
			'EF': ((meta_df['EF_i_kHz'][0] + meta_df['EF_f_kHz'][0])/2) /1e3,
			'e_EF': (np.sqrt(meta_df['EF_i_sem_kHz'][0]**2 + \
					   meta_df['EF_f_sem_kHz'][0]**2)/2) / 1e3,
			'barnu': meta_df['barnu'][0],
			'e_barnu': meta_df['barnu_sem'][0],
			'ff': meta_df['ff'][0],
			'Bfield': meta_df['Bfield'][0],
			'lineshape':lineshape
			}
	
	if isnan(results['EF']):
		results['EF'] = 18/1e3
	results['x_star'] = xstar(meta_df['Bfield'][0], results['EF'])
	results['kF'] = np.sqrt(2*mK*results['EF']*h*1e6)/hbar
	results['omega_over_T'] = results['drive_freq'] / (results['ToTF']*results['EF']* 1e3)
	
	# correct cloud size
	size_names = ['two2D_sv1', 'two2D_sh1', 'two2D_sv2', 'two2D_sh2']
	new_size_names = ['c5_sv', 'c5_sh', 'c9_sv', 'c9_sh']
	for new_name, name in zip(new_size_names, size_names):
		run.data[new_name] = np.abs(run.data[name])

	# average H and V sizes
	run.data['c5_s'] = (run.data['c5_sv']+run.data['c5_sh'])/2
	run.data['c9_s'] = (run.data['c9_sv']+run.data['c9_sh'])/2
	
	# convert delay time to us and add half the pulse length
	run.data['time'] = run.data['time'] * 1000 + \
        (meta_df['trf_us'].values[0]/2.0) 
	
	### DATA FILTERING
	# remove indices if requested
	remove_indices = remove_indices_formatter(meta_df['remove_indices'][0])
	if remove_indices is not None:
		run.data.drop(remove_indices, inplace=True)
	
	# data filtering:
	if FILTER == True:
		# use query in meta_df to filter dataframe, usually to select times
		if not meta_df['query_str'].isna()[0]:
			run.data = run.data.query(meta_df['query_str'][0])
		
	# compute number of total data points
	num = len(run.data)
	
	### CALCULATIONS	
	run.data['detuning'] = run.data['freq'] - meta_df['res_freq'][0] # MHz
	run.data['detuning_EF'] = run.data['detuning']/(results['EF']) # dimless

	# fudge the c9 counts using ff
	run.data['c9'] = run.data['c9'] * results['ff']
	run.data['sum95'] = run.data['c5'] + run.data['c9']
	run.data['ratio95'] = run.data['c9']/run.data['c5']
	run.data['f5'] = run.data['c5']/run.data['sum95']
	run.data['f9'] = run.data['c9']/run.data['sum95']
	
	# NOTE: Frequency source attenuation is freq-dependent,
	# but for these measurements we do spectroscopy over a small freq range.
	# For now, just calculate OmegaR using the average freq.
	avg_freq = run.data[freq_name].mean()
	if meta_df['pulse_type'][0] == 'square':
		pulseareafactor = 1 # np.sqrt(1)
	else:
		pulseareafactor = np.sqrt(0.31)
	OmegaR_dimer = phaseO_OmegaR(meta_df['VVA'].values[0], avg_freq) * pulseareafactor 
	
	if SATURATION_CORRECTION:
		sat_scale_dimer = saturation_scale(OmegaR_dimer**2/(2*np.pi)**2, dimer_x0)
	else:
		sat_scale_dimer = 1
	
	if lineshape == 'conv':
		# load lineshape
		df_ls = pd.read_pickle('./clockshift/convolutions.pkl')
		TTF = round(results['ToTF'], 1)
		if TTF == 0.2:
			TTF = 0.25
		if TTF == 0.7:
			TTF = 0.6
		TRF = results['trf']
		# input is hf/EF
		conv_lineshape = df_ls.loc[(df_ls['TTF']==TTF) & (df_ls['TRF']==TRF)]['LS'].values[0]
		
		# turn convolution lineshape into function
		def conv_func(x, A, x0):
			return A*conv_lineshape(x-x0)/conv_lineshape(0)
	
		lineshape_func = conv_func
	
	elif lineshape == 'sinc2': 
		def sinc2(x, trf):
			"""sinc^2 normalized to sinc^2(0) = 1"""
			t = x*trf
			#print(t)
			return np.piecewise(t, [t==0, t!=0], [lambda t: 1, 
							   lambda t: (np.sin(np.pi*t)/(np.pi*t))**2])
		def sinc2_func(x, A, x0, C):
			x = np.array(x)
			if freq_name == 'freq':
				return A*sinc2((x-x0), results['trf']) + C
			else:
				return A*sinc2((x-x0)*results['EF'], results['trf']) + C
		TTF = results['ToTF']
		lineshape_func = sinc2_func
	elif lineshape == 'gaussian' :
		def gaussian(x, A, x0, sigma, C):
			return A * np.exp(-(x-x0)**2/(2*sigma**2)) + C
		TTF = results['ToTF']
		lineshape_func = gaussian
	elif lineshape == 'gaussian_fixedwidth':
		def gaussian_fixedwidth(x, A, x0, C):
			sigma = 1/results['trf']/results['EF']
			return A* (1/(np.sqrt(2*pi*sigma**2))) * np.exp(-(x-x0)**2/(2*sigma**2)) + C
		TTF = results['ToTF']
		lineshape_func = gaussian_fixedwidth
	num_lineshape_params = len(signature(lineshape_func).parameters)
	
	### MAIN EVALUATION LOOP ###
	times = run.data['time'].unique()
	num_subplots = ceil(np.sqrt(len(times)))
	fig_raw, ax_raw = plt.subplots(num_subplots, num_subplots)
	fig_raw.suptitle(f'Run: {filename}, Spectra and fits at each delay time [raw counts]')
	ax_raw = ax_raw.flatten()
	fig_scl, ax_scl = plt.subplots(num_subplots, num_subplots)
	fig_scl.suptitle(f'Run: {filename}, Spectra and fits at each delay time [Scaled transfer]')
	ax_scl = ax_scl.flatten()
	holding_list = []
	for j, time in enumerate(times):
		sub_df = run.data.query(f'time == {time}').copy()
		time_results_dict = {
			'time':time,
		}
		# why the heck did I do it this way
		time_results_df = pd.DataFrame([time_results_dict])
		for i, (spin, sty) in enumerate(zip(spins, styles)):
					
			mean_counts, max_counts, min_counts =\
				sub_df[spin].mean(), sub_df[spin].max(), sub_df[spin].min()
			if spin != 'ratio95':
				A_guess = -(max_counts - min_counts)
			else:
				A_guess = (max_counts - min_counts)
			if freq_name == 'detuning_EF':
				sigma_guess = 1/results['trf']/results['EF']
			else: 
				sigma_guess = 1/results['trf']
				
			### if freq_name = freq chosen above the spectra fits x axes are B field
			### and resp vs drive plts y axes (left side) are B field 
			### if detuning_EF those axes are detuning / EF   
			if freq_name == 'freq':
				x = BfromFmEB(sub_df[freq_name])
				x0_guess_freq = sub_df.loc[sub_df[spin].idxmin(),freq_name]
				x0_guess = BfromFmEB(x0_guess_freq)
				
			 
				if num_lineshape_params == 5:
					guess = [A_guess, x0_guess, sigma_guess, mean_counts]
				elif num_lineshape_params == 4:
					guess = [A_guess, x0_guess, mean_counts]
					
				try:
					popt, pcov = curve_fit(lineshape_func, x, sub_df[spin], \
								  p0 = guess)
					perr = np.sqrt(np.diag(pcov))
					
					amp = np.abs(popt[0])
					e_amp = perr[0]
					f0 = popt[1]
					e_f0 = perr[1]
					bg = popt[-1]
					e_bg = perr[-1]
				
				except RuntimeError as e:
				# Handle the exception (e.g., print a message or assign default values)
					amp, e_amp, f0, e_f0, bg, e_bg = 0.01,0.01,0.01,0.01,0.01,0.01
					print(f"Error in curve fitting Bfield: {e}, time {time}")
				    # You can assign default values to the variables in case of failure
					

				### the popt[1] used later on to find the FM needs to be the fit from the 
				### detuning_EF not Bfield 
				freq_name_for_FM_etc = 'detuning_EF'
				x_for_FM_etc = sub_df[freq_name_for_FM_etc]
				x0_guess_for_FM_etc = sub_df.loc[sub_df[spin].idxmin(),freq_name_for_FM_etc] 
				
				if num_lineshape_params == 5:
					guess = [A_guess, x0_guess_for_FM_etc, sigma_guess, mean_counts]
				elif num_lineshape_params == 4:
					guess = [A_guess, x0_guess_for_FM_etc, mean_counts]
					
				try: 
					popt_for_FM_etc, pcov_for_FM_etc = curve_fit(lineshape_func, x_for_FM_etc, sub_df[spin], \
							  p0 = guess)
				except RuntimeError as e:
    # Handle the exception (e.g., print a message or assign default values)
					amp, e_amp, f0, e_f0, bg, e_bg = 0.01,0.01,0.01,0.01,0.01,0.01
					print(f"Error in curve fitting frequency for FM etc: {e} at time {time}")
				    # You can assign default values to the variables in case of failure
				    
								
			else:
				x = sub_df[freq_name]
				x0_guess = sub_df.loc[sub_df[spin].idxmin(),freq_name]  ### dim det associated with
																		### first min of counts
				if num_lineshape_params == 5:
					guess = [A_guess, x0_guess, sigma_guess, mean_counts]
				elif num_lineshape_params == 4:
					guess = [A_guess, x0_guess, mean_counts]
				popt, pcov = curve_fit(lineshape_func, x, sub_df[spin], \
							  p0 = guess)
				perr = np.sqrt(np.diag(pcov))
				
				amp = np.abs(popt[0])
				e_amp = perr[0]
				f0 = popt[1]
				e_f0 = perr[1]
				bg = popt[-1]
				e_bg = perr[-1]

			# transfer defined as ratio of fit params
			transfer = amp/bg
			e_transfer = amp/bg * np.sqrt((e_amp/amp)**2 + \
									  (e_bg/bg)**2)
			if CORRECT_SPINLOSS:
				if spin_map(spin) == 'b':
					transfer = transfer/2
					e_transfer = e_transfer/2
			# correct transfer from fit saturation scaling
			# is this only applicable on resonance?...
			transfer = transfer*sat_scale_dimer
			e_transfer = e_transfer * sat_scale_dimer
		
			# turn transfer into GammaTilde
			scaledtransfer = GammaTilde(transfer, h*results['EF']*1e6, OmegaR_dimer*1e3,
							   results['trf']/1e6)
			e_scaledtransfer = e_transfer/transfer * scaledtransfer
			
			sub_df['transfer'] = 1-(sub_df[spin]/bg)
			sub_df['scaledtransfer'] = sub_df.apply(lambda x: GammaTilde(x['transfer'], h*results['EF']*1e6, OmegaR_dimer*1e3,
							   results['trf']/1e6), axis=1)
			if TALK: print(f'Spin: {spin_map(spin)}, Time: {time}, amp: {amp:.1f}+/-{e_amp:.1f},\
				   bg: {bg:.1f}+/-{e_bg:.1f}, transfer: {transfer:.2f}+/-{e_transfer:.2f},\
				   scaledtransfer: {scaledtransfer:.4f}+/-{e_scaledtransfer:.4f}')			
				
			### PLOTTING
			xx = np.linspace(x.min(),x.max(), 100)
			# plot spectra and fits on a nice ol' grid
			ax_raw[j].plot(x, sub_df[spin], label=spin_map(spin), **sty)
			ax_raw[j].plot(xx, lineshape_func(xx, *popt),'--')
			ax_raw[j].set(title=f'{time} us', ylim = [run.data[spins].min().min(), run.data[spins].max().max()])
			
			ax_scl[j].plot(x, sub_df['scaledtransfer'], label=spin_map(spin), **sty)
			yy = lineshape_func(xx, scaledtransfer, popt[1],0)
			ax_scl[j].plot(xx, yy ,'--')
			ax_scl[j].set(title=f'{time} us', ylim=[-0.01, 0.03])
			
			if freq_name == 'freq':
				x_for_FM_etc = sub_df['detuning_EF'] ### to find the correct FM and CS x needs to be this 
										  ### not Bfield 						  
				xx = np.linspace(x_for_FM_etc.min(),x_for_FM_etc.max(), 100)
				popt = popt_for_FM_etc	
				yy = lineshape_func(xx, scaledtransfer, popt[1],0)
			### SAVE RESULTS FOR THIS TIME SPECTRA
			time_results_df['counts_'+spin] = amp
			time_results_df['e_counts_'+spin] = e_amp
			time_results_df['f0_'+spin] = f0
			time_results_df['e_f0_'+spin] = e_f0
			time_results_df['bg_'+spin] = bg
			time_results_df['e_bg_'+spin] = e_bg
			time_results_df['transfer_'+spin] = transfer
			time_results_df['e_transfer_'+spin] = e_transfer
			time_results_df['scaledtransfer_'+spin] = scaledtransfer
			time_results_df['e_scaledtransfer_'+spin] = e_scaledtransfer
			time_results_df['SW_'+spin] = np.trapz(yy, xx)
			time_results_df['e_SW_'+spin] = e_scaledtransfer/scaledtransfer*time_results_df['SW_'+spin]
			time_results_df['FM_'+spin] = np.trapz(yy*xx, xx)
			time_results_df['e_FM_'+spin] = np.abs(e_scaledtransfer/scaledtransfer*time_results_df['FM_'+spin])
			time_results_df['CS_'+spin] = time_results_df['FM_'+spin]/0.5
			time_results_df['e_CS_'+spin] = time_results_df['e_FM_'+spin]/0.5
			time_results_df['Ctilde_'+spin] = time_results_df['CS_'+spin] * pi*kF/(-2*kappa) * (1 + re/a13(results['Bfield'])) 
			time_results_df['e_Ctilde_'+spin] = np.abs(time_results_df['e_CS_'+spin] * pi*kF/(-2*kappa) * (1 + re/a13(results['Bfield']))) 
			if DEBUG:
				print(xx)
				print(yy)
		holding_list.append(time_results_df)
# 		del sub_df
		gc.collect()
	### Finish plotting
	ax_raw[0].legend()
	fig_raw.tight_layout()
	ax_scl[0].legend()
	fig_scl.tight_layout()
	# intermediate df holds the results from all the initial fits
	inter_df = pd.concat(holding_list)
	if SAVE_INTERMEDIATE:
		fig_raw.savefig(os.path.join(interm_folder, 'countfits.png'))
		fig_scl.savefig(os.path.join(interm_folder, 'scaledtransferfits.png'))
		# dump results into pickle
		with open(interm_pkl_file, 'wb') as f:
			pkl.dump(inter_df, f)

	### Calculate contact amplitude and phase shifts
	resp_names = ['counts_', 'SW_', 'FM_', 'CS_', 'Ctilde_', 'Ctilde/counts']
	drive_name = 'f0_'
	num_plots = ceil(np.sqrt(len(resp_names)))

	for i, (spin, sty) in enumerate(zip(spins, styles)):
		fig, ax = plt.subplots(num_plots-1, num_plots)
		ax=ax.flatten()
		fig.suptitle(f'Run: {filename}, Response vs. drive, {spin}')
		for j, resp_name in enumerate(resp_names):
			# First fit peak freq vs. time as proxy for driving Bfield
			bounds = ([0, 0, -10000],
				   [np.inf, 2*np.pi, 10000]
				)
			guess = [inter_df[drive_name+spin].max()-inter_df[drive_name+spin].min(),
				  0, 
				  inter_df[drive_name+spin].mean()]
			popt_drive, pcov_drive = curve_fit(FixedSinkHz, inter_df['time'], 
							  inter_df[drive_name+spin], sigma=inter_df['e_'+drive_name+spin],
								p0=guess, bounds=bounds)
			perr_drive = np.sqrt(np.diag(pcov_drive))
			
			
			# Then fit contact amplitude vs. time 
			bounds = ([0, 0, -10000],
				   [np.inf, 2*np.pi, 10000])
			if resp_name == 'Ctilde/counts':
				resp_name = 'Ctilde_'
				resp_name2 = 'counts_'
				y = inter_df[resp_name+spin]/inter_df[resp_name2+spin]**2 ### this is the y axis on the right of the plt 
									         ### for the orange-y points on c9 plt and blue on c5
			
				sigma = inter_df['e_'+resp_name+spin]/inter_df['e_'+resp_name2+spin]**2
				title = 'Ctilde/counts'
			
			else:
				y = inter_df[resp_name+spin] ### this is the y axis on the right of the plt 
				sigma = inter_df['e_'+resp_name+spin]   ### for the orange-y points and blue on c5
				title = resp_name
				
			guess = [y.max()-y.min(),
			  0, 
			 y.mean()]
			popt_response, pcov_response = curve_fit(FixedSinkHz, inter_df['time'], y, sigma=sigma,
							p0=guess, bounds=bounds)
			perr_response = np.sqrt(np.diag(pcov_response))
			
			xx = np.linspace(0, inter_df['time'].max()+5)
			yy_drive = FixedSinkHz(xx, *popt_drive)
			ax[j].errorbar(inter_df['time'],inter_df[drive_name+spin],
				  yerr=inter_df['e_'+drive_name+spin], mfc='black', mec='black',ecolor='black' )
			ax[j].plot(xx, yy_drive, 'k-') #plotting fit  of drive (black pts)
			ax[j].set(title=title, xlabel='time [us]')
			
			ax2 = ax[j].twinx()
			yy_response = FixedSinkHz(xx, *popt_response)
			ax2.errorbar(inter_df['time'], y , yerr=sigma, **sty)
			ax2.plot(xx, yy_response, '--')
			
			results['PS_' + resp_name+spin] = popt_response[1] - popt_drive[1]
			results['e_PS_' + resp_name + spin] = np.sqrt(perr_response[1]**2 + perr_drive[1]**2)
			results['AMP_'+resp_name+spin] = popt_response[0]
			results['e_AMP_' + resp_name + spin] = np.sqrt(perr_response[0]**2)
			
			
		
		fig.tight_layout()
		if SAVE_INTERMEDIATE:
			fig.savefig(os.path.join(interm_folder, f'response_vs_drive_{spin}.png'))
	
###############################
####### Saving Results ########
###############################
	
	if SAVE_RESULTS == True:
		savedf = pd.DataFrame(results, index=[save_df_index])
		save_df_index += 1
		save_df_row_to_xlsx(savedf, savefile, filename)
		
	results_list.append(results)
	
	# dump results into pickle
	with open(pkl_file, 'wb') as f:
	        pkl.dump(results_list, f)

###############################
###### Summary Plotting #######
###############################	
	
if PLOT_SUMMARY:
	# plotting options
	# plot_options = {"SR Fudge":False,
	# 				"Compare SW and FM": False,
	# 				"Loss Contact": False,
	# 				"Open Channel Fraction": True,
	# 				"Sum Rule 0.5": True}
	
	true_options = []
	# for key, val in plot_options.items():
	# 	if val:
	# 		true_options.append(key)
				
	title_end = ' , '.join(true_options)
	if title_end != '':
		title_end = " with " + title_end
	plot_title = "Four shot analysis" + title_end
	
	# get all keys form recent results dict, make a new summary dict
	#keys = results.keys()	
	keys = results_list[0].keys()
	values = [[result[key] for result in results_list] for key in keys]	
	summary = dict(zip(keys, values))
	
	def summary_plotter(prefix_str, x_name, resp_names, ylims=None):
		num_plots = ceil(np.sqrt(len(resp_names)))
		fig, ax = plt.subplots(num_plots-1, num_plots)
		ax = ax.flatten()
		for i, (spin, sty) in enumerate(zip(spins, styles)):
			for j, resp_name in enumerate(resp_names):
				x = summary[x_name]
				y = summary[prefix_str + resp_name + spin]
				yerr = summary['e_' + prefix_str + resp_name + spin]
				ax[j].errorbar(x, y, yerr=yerr, label=spin_map(spin), **sty)
				ax[j].set(xlabel=x_name, ylabel=resp_name[:-1], ylim=ylims)
		ax[0].legend()
		fig.suptitle(f'{prefix_str} vs {x_name}')
		fig.tight_layout()
		if SAVE_RESULTS: fig.savefig(os.path.join(proj_path, f'figures/{prefix_str}vs_{x_name}.png'))
		return
	
	resp_names = ['amp_', 'SW_', 'FM_', 'CS_', 'Ctilde_']
	summary_plotter('PS_', 'drive_freq', resp_names)
	summary_plotter('AMP_', 'drive_freq', resp_names)
	summary_plotter('PS_', 'ToTF', resp_names)
	summary_plotter('AMP_', 'ToTF', resp_names)
	summary_plotter('PS_', 'omega_over_T', resp_names, ylims=[0, 0.6])
	summary_plotter('AMP_', 'omega_over_T', resp_names)