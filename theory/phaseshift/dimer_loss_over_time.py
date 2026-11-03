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
from fit_functions import Gaussian, Sinc2, Sin2Decay
from library import FreqMHz
from data_class import Data
import matplotlib.colors as mc
import colorsys

def sin2dec(x, A, omega, phi, C, tau):
    return A*np.exp(-x/tau)*np.sin(omega*x - phi)**2  + C
def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

run = '2024-04-24_K'
run_fn = run + '_e.dat'

x_name = "time"
y_name = "sum95"

guess = [5000, 0.1, 0, 20000, 20]  # A, omega, phi, C, tau
data_folder = 'data/'

data = Data(run_fn, path=data_folder)
# data.fit(Sin2Decay, names=[x_name, y_name], guess=guess)
df = data.data.groupby('time').agg({y_name:['mean','std','sem']}).reset_index()
popt,pcov = curve_fit(sin2dec, df.time, df[y_name,'mean'], p0= guess)

xx = np.linspace(df.time.min(), df.time.max(), 200)
yy = sin2dec(xx, *popt)

plt.rcParams.update({'font.size': 14})
color='black'
fig, ax =plt.subplots()
ax.errorbar(df.time, df[y_name]['mean'], yerr = df[y_name]['std'], marker='o', markersize=6, ls='none', \
             mfc = 'white', mec =adjust_lightness(color), mew=2, ecolor=color, capsize=2)
ax.plot(xx, yy, color=color,label='$A*\exp(-t/ \\tau)*\sin(\omega t - \phi)^2 + C$')
plt.setp(ax.spines.values(), linewidth=2)
ax.tick_params(width=2)

ax.set_ylabel('Atom num [arb.]')
ax.set_xlabel('Time [us]')
ax.legend()
plt.tight_layout()
# plt.savefig('dimer_loss_over_time.png', dpi=600)
# plt.savefig('dimer_loss_over_time.pdf', format='pdf', dpi=600, bbox_inches='tight')
