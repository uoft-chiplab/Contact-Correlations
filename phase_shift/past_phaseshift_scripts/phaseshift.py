# -*- coding: utf-8 -*-
"""
2023-10-19
@author: Chip Lab

script to analyse phase shift measurements

Relies on data_class.py
"""

from data_class import * 
from wiggle_cal import * 

# delay_times = np.array(np.linspace(0,0.77,19))
delay_times_new = [0,0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.4,0.44,0.48,0.52,0.56,0.6,0.64,0.68]
delay_times_new = [0,0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.4,0.44,0.48,0.52,0.56,0.6,0.64,0.68,0.72,0.76]

delay_times = np.array(np.linspace(0.005,0.565,15))

names1 = ['freq','sum95']
names = ['freq','G_ToTFcalc']

directory1 = os.fsencode('E:/Analysis Scripts/analysis/data/2023-11-14_F/')
directory = os.fsencode('E:/Analysis Scripts/analysis/data/2023-11-15_C/')


num = 100
x_list = np.linspace(0, 0.6, num)

# only values lists and excluding 0.41 

def plots(func, directory):
	amp_list=np.array([])
	amper_list=np.array([])
	x0_list=np.array([])
	x0er_list=np.array([])	 
	sigma_list=np.array([])	 
	
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if directory == directory1:
			names = names1		
			if func == MinSinc2:
				if filename.endswith("0.41.dat"):
					data = Data(filename, column_names=names)
					data.fitnoplots(func,names=names,guess=[-8000,43.20,0.06,28000])
				else:
					data = Data(filename, column_names=names)
					data.fitnoplots(func,names=names)
			if func == MinFixedSinc2:
				if filename.endswith("0.53.dat"):
					data = Data(filename, column_names=names)
					data.fitnoplots(func,names=names,guess=[-6000,43.23,26000])
				else:
					data = Data(filename, column_names=names)
					data.fitnoplots(func,names=names)
			else:
				data = Data(filename, column_names=names)
				data.fitnoplots(func,names=names)
			amp = data.popt[0]
			amp_list = np.append(amp_list,amp)
			errors = np.sqrt(np.diag(data.pcov))
			amper = errors[0]
			amper_list = np.append(amper_list,amper)
			x0 = data.popt[1]
			x0_list = np.append(x0_list,x0)
			x0er = errors[1]
			x0er_list = np.append(x0er_list, x0er)
			sigma = data.popt[2]
			sigma_list = np.append(sigma_list, sigma)
		else:
			data = Data(filename, column_names=['freq','ToTFcalc'])
			data.fitnoplots(func,names=['freq','ToTFcalc'])
			amp = data.popt[0]
			amp_list = np.append(amp_list,amp)
			errors = np.sqrt(np.diag(data.pcov))
			amper = errors[0]
			amper_list = np.append(amper_list,amper)
			x0 = data.popt[1]
			x0_list = np.append(x0_list,x0)
			x0er = errors[1]
			x0er_list = np.append(x0er_list, x0er)
			sigma = data.popt[2]
			sigma_list = np.append(sigma_list, sigma)
		
	return amp_list, amper_list, x0_list, x0er_list, sigma_list


def plotamp(func,directory):
	fig = plt.figure(0)
	
	fit_func, guess, params = FixedSin(np.array(list(zip(delay_times_new,plots(func,directory)[0]))), 2.5)

	popt2, pcov2 = curve_fit(fit_func, delay_times_new, plots(func,directory)[0], p0=guess, sigma=plots(func,directory)[1])
	perr2 = np.sqrt(np.diag(pcov2))

	plt.title('{} Fit'.format(func))
	plt.xlabel('delay time (ms)')
	plt.ylabel('amplitude')

	plt.errorbar(delay_times, B_list/popt[2]*np.average(plots(func,directory)[0]), yerr=B_err_list, fmt='o',label='wiggle cal')

	plt.errorbar(delay_times_new,plots(func,directory)[0],yerr=plots(func,directory)[1],fmt='o', label='heating meas')
	plt.plot(x_list, fit_func(x_list, *popt2), 'g')

	plt.legend()
	
	print(*popt2, perr2)

	return fig

def plotx0(func,directory):
	fig = plt.figure(0)
	
	fit_func, guess, params = FixedSin(np.array(list(zip(delay_times_new,plots(func,directory)[2]))), 2.5)

	popt2, pcov2 = curve_fit(fit_func, delay_times_new, plots(func,directory)[2], p0=guess, sigma=plots(func,directory)[3])
	perr2 = np.sqrt(np.diag(pcov2))

	plt.title('{} Fit'.format(func))
	plt.xlabel('delay time (ms)')
	plt.ylabel('freq center (MHz)')
	plt.errorbar(delay_times, B_list/popt[2]*np.average(plots(func,directory)[2]), yerr=B_err_list, fmt='o',label='wiggle cal')

	plt.errorbar(delay_times_new,plots(func,directory)[2],yerr=plots(func,directory)[3],fmt='o',label='heating meas')
	
	plt.plot(x_list, fit_func(x_list, *popt2), 'g')

	plt.legend()
	
	print(*popt2, perr2)
	
	return fig

#outputting lists and fit plots and tables 

def plotsplots(func,directory):
	amp_list=np.array([])
	amper_list=np.array([])
	x0_list=np.array([])
	x0er_list=np.array([])	 
	
	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename.endswith(".dat"):
			data = Data(filename, column_names=names1)
			try:	
				data.fit(func,names=names1)
			except RuntimeError:
				print('failed to fit {}'.format(filename))
			continue 
			amp = data.popt[0]
			amp_list = np.append(amp_list,amp)
			errors = np.sqrt(np.diag(data.pcov))
			amper = errors[0]
			amper_list = np.append(amper_list,amper)
			x0 = data.popt[1]
			x0_list = np.append(x0_list,x0)
			x0er = errors[1]
			x0er_list = np.append(x0er_list, x0er)
			
	return amp_list, amper_list, x0_list, x0er_list

# B_list = np.array(list(map(B_from_FreqMHz, freq_list))).flatten()
# B_list_plus = np.array(list(map(B_from_FreqMHz, freq_list+freq_err_list)))
# B_list_minus = np.array(list(map(B_from_FreqMHz, freq_list-freq_err_list)))

# calculate error in B field, 
# by checking +freq and -freq error, taking the largest
# B_err_list = np.array([])
# for i in range(len(B_list)):
# 	B_err = max([np.abs(B_list[i]-B_list_plus[i]),
# 			  np.abs(B_list[i]-B_list_minus[i])])
# 	B_err_list = np.append(B_err_list, B_err)
# 	

# fit_func, guess, params = FixedSin(np.array(list(zip(delay_times,B_list))), 2.5)
# 	
# popt, pcov = curve_fit(fit_func, delay_times, B_list, p0=guess, sigma=B_err_list)
# perr = np.sqrt(np.diag(pcov))


# plt.figure()
# plt.errorbar(delay_times, B_list, yerr=B_err_list, fmt='o')
# # plt.plot(x_list, fit_func(x_list, *guess), 'r--')
# plt.plot(x_list, fit_func(x_list, *popt), 'g')
# plt.xlabel("Delay Time (ms)")
# plt.ylabel("Magnetic Field B (G)")
# plt.ylim(202, 202.2)
# plt.show()

# print(*popt)
# print(*perr)

# for i in range(len(delay_times)):
# 	data = Data(file).data.loc[Data.data['delay']==delay_times[i]]
# 	data.popt, data.pcov = curve_fit(fit_func, data.data['freq'], 
# 						data.data['fraction95'])
# 	data.err = np.sqrt(np.diag(data.pcov))
# 	
# 	freq_list = freq_list.append(data.popt[1])
# 	
# 	uncert_list = uncert_list.append(data.err[1])
# 	
# 	
# 	data_list = data_list.append(data)
# 	
# 	print(data_list)