# -*- coding: utf-8 -*-
"""
2025-03-20

@author: Chip Lab
"""

from contact_correlations.UFG_analysis import BulkViscTrap
from library import styles, colors

import numpy as np
import matplotlib.pyplot as plt

ToTFs = [0.25, 0.55]
EFs = [15e3, 20e3]

# fix this
barnu = 309 * np.sqrt(1.5)
nus = np.array([0])

# initialize plots
fig, axes = plt.subplots(2,2, figsize=(8,8))
axs = axes.flatten()

ynames = [
		r'Density averaged density $n_\sigma\lambda^3$', 
		  r'Density averaged density $n_\sigma$ [1/cm$^3$]',
		  r'Peak density $k_F^3/6\pi^2$ [1/cm$^3$]',
		  '',
		  ]
xname = r'$T/T_F$'

# add plot settings to axs
for j, ax in enumerate(axs):
	ax.set(xlabel=xname, ylabel=ynames[j])

# loop over temps and compute trapped gas properties
# for i in range(len(ToTFs)):
# 	ToTF = ToTFs[i]
# 	EF = EFs[i]
# 	
# 	BVT = BulkViscTrap(ToTF, EF, barnu, nus)
# 	print(ToTF)
# 	print(BVT.ns)

num = 50
EF = 15e3
xs = np.linspace(ToTFs[0], ToTFs[-1], num)
BVTs = [BulkViscTrap(x, EF, barnu, nus) for x in xs]

ax = axs[0]
ys = np.array([BVT.ns for BVT in BVTs])
ax.plot(xs, ys, color=colors[0])

ax = axs[1]
ys = np.array([BVT.ns/(BVT.lambda_T**3*10**6) for BVT in BVTs])
ax.plot(xs, ys, color=colors[0])

ax = axs[2]
ys = np.array([BVT.kF**3/(6*np.pi**2) for BVT in BVTs])
ax.plot(xs, ys, color=colors[0])

fig.tight_layout()
plt.show()
	