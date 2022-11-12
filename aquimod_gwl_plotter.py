# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:49:50 2022

@author: kieran.white

Notes

Change filepath variable names to something more general.

Fitting Weibull distribution to simulated groundwater levels needs to be
compared to simply constructing cumulative probability distribution from the
data and taking its median value.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

# Section checking clustering of parameters, and extracts number of accepted
# simulations.

# Aquifer storage coefficient, 'S'.
Q1K1S1path = "C:\\Users\\KIERAN.WHITE\\Documents\\AquiMod2\\127479\\Outpu" \
    "t\\Q3K3S1_calib.out"
Q1K1S1data = np.genfromtxt(Q1K1S1path, encoding = "utf-8-sig", dtype = float)
S = Q1K1S1data[1:, 1]

# Get number of accepted simulations, 'N'.
N = len(S)

# Weibull distribution shape parameter, 'k'.
weibullpath = "C:\\Users\\KIERAN.WHITE\\Documents\\AquiMod2\\127479\\Outp" \
    "ut\\Weibull_calib.out"
weibulldata = np.genfromtxt(weibullpath, encoding = "utf-8-sig", dtype = float)
k = weibulldata[1:, 0]

# Objective function, 'OF'.
fitpath = "C:\\Users\\KIERAN.WHITE\\Documents\\AquiMod2\\127479\\Output\\" \
    "fit_calib.out"
fitdata = np.genfromtxt(fitpath, encoding = "utf-8-sig", dtype = float)
OF = fitdata[1:]
###

# Setup variables to simplifying reading and looping through paths for storing
# each accepted simulation. An accepted simulation is where a model resulting
# from Monte Carlo sampling of parameters is above the acceptable model
# threshold set in the Input.txt file when running calibration.
Name_of_runs = "C:\\Users\\KIERAN.WHITE\\Documents\\AquiMod2\\127479\\Out" \
    "put\\Q3K3S1_TimeSeries"
Extension_of_runs = ".out"

# Initialise dictionary for storing each accepted simulation.
d = {}

# Loop to store column values, ignoring headers. Iterable 'i' acts as
# incrementer and dictionary key. Key 'i' stores the results of accepted
# simulation 'i + 1'.
for i in range(N):
    path = Name_of_runs + str(i + 1) + Extension_of_runs
    data = np.genfromtxt(path, encoding = "utf-8-sig", dtype = float)
    d[i] = data[1:, :]

# Get number of sampling points, 'N_t'.
N_t = len(d[0][:, 0])

# Initialise dictionary for storing ground water level values for each accepted
# simulation, 'd_wgl'.
d_gwl = {}
# Initialise array for storing median simulated ground water levels,
# 'sim_gwl_med'.
sim_gwl_med = np.zeros(N_t)
# Initialise arrays for storing top and bottom bounds of confidence interval, 
# 'ci_top' and 'ci_bot'. Confidence interval 'ci' set.
ci = 0.99
ci_top = np.zeros(N_t)
ci_bot = np.zeros(N_t)

# Fill initialised arrays for data visualisation.
for k in range(N_t):
    x = np.zeros(N)
    for i in range(N):
        x[i] = d[i][k, len(d[0][0, :]) - 1]
    d_gwl[k] = x
    shape, loc, scale = ss.weibull_min.fit(x, floc=0)
    sim_gwl_med[k] = ss.weibull_min.median(shape, loc, scale)
    ci_bot[k], ci_top[k] = ss.weibull_min.interval(ci, shape, loc, scale)


# Section for checking normalised frequency histogram.
N_res = 1000
barx = np.zeros(N_res)
for i in range(0, N_res):
    barx[i] = 0.5 + i

a1 = np.zeros(N_res)
for i in range (0, N):
       
       for n in range(0, N_res):
              
              if x[i] >= (1 * n) + 0 and x[i] < (1 * (n + 1)) + 0:
                     a1[n] += 1
              else:
                     pass

a1_norm = a1 / N

#(N / N_res)

a1_cdf = np.zeros(N_res)

for i in range(N_res):
    a1_cdf[i] = sum(a1_norm[0:i])
"""
plt.bar(barx, a1_cdf, width = 1.0, color = 'b')
"""
# Load observation data for ground water levels.
obs_path = "C:\\Users\\KIERAN.WHITE\\Documents\\AquiMod2\\127479\\Observatio" \
    "ns.txt"
obs_data = np.genfromtxt(obs_path, encoding = "utf-8-sig", delimiter='\t', 
                         dtype = float)
obs_gwl = obs_data[3:, 6]

x_pl = np.arange(0, k + 1, 1)
"""
# Collect median simulated ground water level at each time point.
sim_gwl_med = np.zeros(N_t)
for i in range(N_t):
    sim_gwl_med[i] = ss.weibull_min.median(shape, loc, scale)

shape, loc, scale = ss.weibull_min.fit(x, floc=0)
"""
#ss.weibull_min.interval(0.9, shape, loc, scale)
"""
barx = np.zeros(100)
for i in range(0, 100):
    barx[i] = 0.5 + i

x1 = np.arange(0, 100, 1)
plt.plot(x1, ss.weibull_min.pdf(x1, shape, loc, scale), color = 'r')

x_pl = np.arange(0, k + 1, 1)

plt.fill_between(x_pl, ci_top, ci_bot, color = 'y')

plt.plot(x_pl, sim_gwl_med, color = 'k')

plt.scatter(x_pl, obs_gwl, s=0.2, color = 'c')
"""
