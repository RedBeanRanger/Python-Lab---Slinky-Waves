# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:58:35 2021

@author: angel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import e

pointX, amplitude = np.loadtxt(r"C:\Users\angel\Desktop\Docs\PHY244\SlinkyWaves\SlinkyWavesExercise5.txt", 
                               unpack = True, skiprows = 1)

distanceX = pointX * 30 # convert to distances in cm, with an error of errX
distanceX = np.array(distanceX)
errX = 0.1*np.ones(len(distanceX))

distanceX = np.array(distanceX)
amplitude = np.array(amplitude)
errY = errX # error in X and Y are the same in this case

plt.errorbar(distanceX, amplitude, xerr = errX, yerr = errY, marker = "o", ls = "none", 
             label = "Distance Versus Amplitude")
plt.title("Distance Along the Slinky versus Amplitude of Oscillation")
plt.xlabel("Distance (cm)")
plt.ylabel("Amplitude of Oscillation (cm)")
plt.legend()

def model_amplitude_single_exponential(dist, k, c):
    # models amplitude with a single exponential function
    return c*e**(-k*dist)

def model_amplitude_double_exponential(dist, k_1, k_2, c_1, c_2):
    # models amplitude as a sum of exponential functions
    return c_1*e**(-k_1*dist) + c_2*e**(k_2*dist)

# Find optimal parameters
popt, pcov = curve_fit(model_amplitude_single_exponential, distanceX, amplitude, p0 = (-1/180, -180))
popt2, pcov2 = curve_fit(model_amplitude_double_exponential, distanceX, amplitude,
                         p0 = (-1/180, 5, 0, 0))

model_curve_single_exp = []
model_curve_double_exp = []


print(popt2)

for i in distanceX:
    model_curve_single_exp.append(model_amplitude_single_exponential(i, popt[0], popt[1]))
    model_curve_double_exp.append(model_amplitude_double_exponential(i, popt2[0], popt2[1], popt2[2], popt2[3]))
    
# Reduced chi squared calculation
def redChiSquared(measured_data, predicted_data, err):
    # returns chiSquared value
    # takes in arrays of measured_data, predicted_data, and err
    # assume the arrays have the same length
    sum = 0
    for x in range(0, len(measured_data)):
        #sum += ((10**(measured_data[x]) - 10**(predicted_data[x])) / 10**(err[x]))**2
        sum += (((measured_data[x]) - (predicted_data[x])) / (err[x]))**2
    return sum/(len(measured_data) - 2) # curve fitting 2 parameters

print("The reduced chi squared value of the single exponential is: ", redChiSquared(amplitude, model_curve_single_exp, errY))
print("The reduced chi squared value of the sum of exponentials is: ", redChiSquared(amplitude, model_curve_double_exp, errY))

plt.errorbar(distanceX, model_curve_single_exp, linestyle = "-", label = "Single Exponential")
plt.errorbar(distanceX, model_curve_double_exp, linestyle = "-", label = "Sum of Exponentials")
plt.legend()
plt.show()