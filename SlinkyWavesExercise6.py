# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:58:39 2021

@author: angel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import e

pointX, amplitude = np.loadtxt(r"C:\Users\angel\Desktop\Docs\PHY244\SlinkyWaves\SlinkyWavesExercise6.txt", 
                               unpack = True, skiprows = 1)

distanceX = pointX[1:] * 30 # convert to distances in cm, with an error of errX
distanceX = np.array(distanceX)
errX = 0.1*np.ones(len(distanceX))

distanceX = np.array(distanceX)
amplitude_modified = np.array(amplitude[1:])
errY = errX # error in X and Y are the same in this case

plt.errorbar(distanceX, amplitude_modified, xerr = errX, yerr = errY, marker = "o", ls = "none", 
             label = "Distance Versus Amplitude")
plt.title("Distance Along the Slinky versus Amplitude of Oscillation")
plt.xlabel("Distance (cm)")
plt.ylabel("Amplitude of Oscillation (cm)")
plt.legend()

# Model with linear curve
def model_amplitude(dist, k, c):
    # models amplitude as a linear function of distance and some constant, k,
    # offset by some c
    return (dist + c)*k

def model_amplitude_with_exponential(dist, k, c):
    # model amplitude as an exponential function of distance with some constants k, c
    return c*e**(dist*k)

# Find optimal parameters for linear curvefit
popt, pcov = curve_fit(model_amplitude, distanceX, amplitude_modified, p0 = (-1/180, -180))
print("Distance is proportional to amplitude by a factor of: ", popt[0])
print("Offset is: ", popt[1])

# Find optimal parameters for nonlinear curvefit
popt2, pcov2 = curve_fit(model_amplitude_with_exponential, pointX*30, amplitude, p0 = (4, -1))

# Construct Linear Model
model_curve = []
for i in distanceX:
    model_curve.append(model_amplitude(i, popt[0], popt[1]))

# Construct Nonlinear Model
model_curve2 = []
for i in pointX*30:
    model_curve2.append(model_amplitude_with_exponential(i, popt2[0], popt2[1]))

#Curve with theoretical y_d:
theoretical_curve = [model_amplitude(0, popt[0], popt[1])]
theoretical_curve.extend(model_curve)
theoretical_x = [0]
theoretical_x.extend(distanceX)


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

print("The reduced chi squared value of the linear fit is: ", redChiSquared(amplitude_modified, model_curve, errY))
print("The reduced chi-square of the nonlinear fit is: ", redChiSquared(amplitude, model_curve2, 0.1*np.ones(7)))

plt.errorbar(distanceX, model_curve, linestyle = "-", label = "Curve fit")
plt.errorbar(pointX*30, model_curve2, linestyle = "-", label = "Non-linear Curve_fit")
plt.plot(0, amplitude[0], linestyle = "none", marker = "o", label = "Eliminated Data")
plt.plot(0, theoretical_curve[0], linestyle = "none", marker = "o", label = "Theoretical y_d")
plt.plot(theoretical_x, theoretical_curve, linestyle = "--")
plt.legend()
plt.show()