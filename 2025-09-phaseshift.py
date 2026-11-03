# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 2025 to analyize new phase shift measurements 

@author: coldatoms
"""
# paths
import os
import sys

home = os.path.abspath(__file__ + 6 * '/..')
data_file = os.path.join(home, 'Data\\2025\\09 September2025\\13September2025\\C_phaseshift_scan_6kHz_10uspulse')
analysis_path = os.path.join(home, 'Analysis Scripts\\analysis')
if analysis_path not in sys.path:
	sys.path.insert(0, analysis_path)
from data_class import Data
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
from scipy.optimize import curve_fit
import re

filename = '2025-09-13_C'

grabbing_times = glob.glob(os.path.join(data_file, f'{filename}_e_time=*.dat'))
times = [float(re.search(r'time=([0-9.]+)(?=\.dat)', f).group(1)) for f in grabbing_times]

def doublesinc2(x,A,x0,p,B,x1,q,C):
      return A*np.sinc((x-x0)/p)**2 + B*np.sinc((x-x1)/q)**2 + C

def sinc2(x, A, x0, p, C):
      return A*np.sinc((x-x0)/p)**2 + C

pulse_time = [
      10, #us
]

time_list = times 
# [
#       0.01, #ms
#              0.02, 
#              0.03,
#              0.04,
#              0.05
#              ]

f0s = []
f0s_202 = []
As = []
for time in time_list:
    run = Data(f'{filename}_e_time={time}.dat')
    df = run.data

    bg = df[df['vva'] == 0]
    data = df[df['vva'] != 0 ]
#     data['N'] = data['c5'] - bg['c5'].mean() + data['c9']
#     data['transfer'] = (data['c5'] - bg['c5'].mean())/(data['N'])
    avgdata = df.groupby('freq').mean().reset_index()
    bg = avgdata[avgdata['vva'] != 4]
    avgdata = avgdata[avgdata['vva'] == 4]
    if bg['c5'].empty:
          bg['c5'] = pd.Series([0])
    avgdata['N'] = avgdata['c5'] - bg['c5'].mean() + avgdata['c9']
    avgdata['transfer'] = (avgdata['c5'] - bg['c5'].mean())/(avgdata['N'])
    avgdata['loss'] = (bg['c9'].mean() - avgdata['c9'])/bg['c9'].mean()

    yname = 'transfer'
    xname = 'freq'

    #finding where the dips are to try and fit them generically 
    ymin1 = avgdata[avgdata[xname] < 43.2][yname].nsmallest(1).index[0]
    ymin2 = avgdata[avgdata[xname] > 43.2][yname].nsmallest(1).index[0]

    xmin1 = avgdata.loc[ymin1, xname]
    xmin2 = avgdata.loc[ymin2, xname]
    
    #amplitudes 
    amp = avgdata[yname].max() - avgdata[yname].min()
    #offset 
    offset = avgdata[yname].max()

    popt, pcov = curve_fit(doublesinc2, avgdata[xname], avgdata[yname], p0=[-amp,xmin1,0.1,-amp,xmin2,.1,offset])
    try:
         popt2, pcov2 = curve_fit(sinc2, avgdata[avgdata[xname] < 43.1][xname], avgdata[avgdata[xname] < 43.1][yname], p0=[-amp,xmin1,0.100, offset])
    except RuntimeError:
          xmin1 = popt[1]
    xs = np.linspace(avgdata[xname].min(), avgdata[xname].max(), 100)
    xs2 = np.linspace(avgdata[avgdata[xname] < 43.2][xname].min(), avgdata[avgdata[xname] < 43.2][xname].max(), 100)
    
    fig, ax = plt.subplots()

    ax.plot(avgdata[xname], avgdata[yname])
    ax.plot(xs, doublesinc2(xs, *popt), marker='', linestyle='-', color='blueviolet')
    ax.plot(xs2, sinc2(xs2, *popt2), marker='', linestyle='-', color='mediumvioletred')
    ax.set(
          ylabel = yname,
          xlabel = 'Frequency (MHz)',
          title = f'{time*1000} us'
    )

    f0 = popt[1]
    f0s.append(f0)
    f0_202 = popt[4]
    f0s_202.append(f0_202)
    A = popt[0]
    As.append(A)

results_df = pd.DataFrame({'f0s':f0s,
                           'f0s_202':f0s_202,
                           'time':time_list})

# fit to sinusoid
ts = results_df["time"]
f0s = results_df['f0s']
fig, ax = plt.subplots(2)

x_osc_time = np.linspace(ts.min(), ts.max(),100)
y_osc_time = np.linspace(f0s.min(), f0s.max(), 100)
def fixedsin(x, a, p, b):
      w = 6*(2*np.pi)
      return a*np.sin(w*(x+p)) + b
def sin(x, a, w, p, b):
      return a*np.sin(w*(x+p)) + b
p0=[0.1, 100, 0.2*np.pi, 43.3]
popt, pcov = curve_fit(sin, ts, f0s, p0, bounds = ([0, 0, 0, 0], [1, np.inf, 2*np.pi, np.inf]))
popt_fixed, pcov = curve_fit(fixedsin, ts, f0s, [0.1, 100, 43.3], bounds = ([0, 0, 0], [1, np.inf, np.inf]))

ax[0].plot(x_osc_time, sin(x_osc_time, *popt), ls="-", marker="", color="plum")
ax[0].plot(x_osc_time, fixedsin(x_osc_time, *popt_fixed), ls="-.", marker="", color="mediumorchid", label='Fixed 6 kHz')

ax[0].plot(ts, f0s, color="hotpink", label=f"a={popt[0]:.2f}, w={popt[1]/(2*np.pi):.2f}kHz, p={popt[2]/np.pi:.2f} pi, b={popt[3]:.2f}")
ax[0].set(
      ylabel = 'Frequency centers from fit [MHz]',
      # xlabel = 'Time (ms)'
)
ax[0].legend()

# popt_amp, pcov_amp = curve_fit(sin, ts, As)
popt_amp_fixed, pcov_amp = curve_fit(fixedsin, ts, As)
# ax[1].plot(x_osc_time, sin(x_osc_time, *popt_amp), marker='', linestyle='-', color='cadetblue')
ax[1].plot(x_osc_time, fixedsin(x_osc_time, *popt_amp_fixed), marker='', linestyle='-.', color='powderblue')
ax[1].plot(ts, As, color='darkturquoise')
ax[1].set(
      ylabel = 'Amplitude',
      xlabel = 'Time (ms)'
)

fig.tight_layout()