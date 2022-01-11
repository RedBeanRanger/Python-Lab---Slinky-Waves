# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:49:22 2021

@author: angel
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

w2, k2, error_w2, error_k2, delta_x = np.loadtxt(r"C:\Users\angel\Downloads\SlinkyWavesExercise3.txt", 
                                        skiprows=1, unpack = True)

"""
w2 = data[:,0]
k2 = data[:,1]
error_w2 = data[:,2]
error_k2 = data[:,3]
"""

plt.errorbar(w2, k2, yerr=error_k2, xerr=error_w2, marker='o', ls='none', label='data')
plt.xlabel('w2')
plt.ylabel('k2')
plt.title('w2 vs. k2')
plt.legend()

def k_2(measured_w2, w0, c0):
    return (measured_w2 - w0**2)/(c0**2)

popt, pcov = curve_fit(k_2, xdata = w2, ydata = k2, sigma = error_k2, absolute_sigma = True,
                       p0 = [86, 1.51])

print("w0 is:", popt[0])
print("c0 is:", popt[1])

model_k2 = []
for i in w2:
    model_k2.append(k_2(i, popt[0], popt[1]))
    
plt.errorbar(w2, model_k2)
plt.savefig('w2 vs. k2 plot.pdf')