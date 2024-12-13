import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

h = 6.62607015e-34  # Planck's constant (J*S)
c = 3e9  # Speed of light (m/s)
Kb = 1.380699e-23  # Boltzmann constant (J/K)

def blackbody(x, T):
    numerator = (8 * np.pi * h * c) / ((x * 1e-9)**5)
    denominator = 1 / (np.exp(h * c / (x * 1e-9 * Kb * T)) - 1)
    spectrum = numerator * denominator
    return spectrum / np.sum(spectrum)

def fit_and_extract(func, x, y, p0):
    popt, pcov = curve_fit(func, x, y, p0=p0)
    return popt[0], np.sqrt(np.diag(pcov))[0]

# Initialize arrays for temperature and uncertainty
T = np.zeros(6)
Tunc = np.zeros(6)
V = np.array([3, 4, 5, 6, 7, 8])

# Filenames for different voltages
filenames = [
    'C:/Users/lando/PycharmProjects/PHYS-270L/BlackbodyRad-Analysis/BlackBody Spectrum/BlackBody Spectrum/3.0V00000.txt',
    'C:/Users/lando/PycharmProjects/PHYS-270L/BlackbodyRad-Analysis/BlackBody Spectrum/BlackBody Spectrum/4.0V00000.txt',
    'C:/Users/lando/PycharmProjects/PHYS-270L/BlackbodyRad-Analysis/BlackBody Spectrum/BlackBody Spectrum/5.0V00000.txt',
    'C:/Users/lando/PycharmProjects/PHYS-270L/BlackbodyRad-Analysis/BlackBody Spectrum/BlackBody Spectrum/6.0V00000.txt',
    'C:/Users/lando/PycharmProjects/PHYS-270L/BlackbodyRad-Analysis/BlackBody Spectrum/BlackBody Spectrum/7.0V-25ms00000.txt',
    'C:/Users/lando/PycharmProjects/PHYS-270L/BlackbodyRad-Analysis/BlackBody Spectrum/BlackBody Spectrum/8.0V-20ms00000.txt'
]

# Process data for each voltage
for i, filename in enumerate(filenames):
    lamda, counts = np.genfromtxt(filename, usecols=(0, 1), skip_header=17, skip_footer=1, unpack=True)
    counts /= np.sum(counts)  # Normalize counts

    T[i], Tunc[i] = fit_and_extract(blackbody, lamda, counts, p0=3000)

    # Plot the spectrum and fit
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(lamda, counts, linestyle='solid', color='gold', label=f'{V[i]}V Measured Spectrum')
    plt.plot(lamda, blackbody(lamda, T[i]), linestyle='dashed', color='purple', label=f'{V[i]}V Fit Spectrum')
    plt.yscale("log")
    plt.ylim(1e-4, 1)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (Counts)')
    plt.title(f'{V[i]}V Light Bulb Spectrum')
    plt.legend()
    plt.show()

# Plot Temperature vs Voltage
plt.figure(figsize=(7, 5), dpi=300)
plt.errorbar(V, T, yerr=Tunc, fmt='o', color='purple', label='Measured Data')
plt.xlabel('Bulb Voltage (V)')
plt.ylabel('Filament Temperature (K)')
plt.title('Temperature vs. Bulb Voltage')
plt.legend(loc=2)
plt.show()

# CMB Spectrum

def BlackbodyFreq(x, T):
    numerator = (8 * np.pi * h * (x * 3e10)**3) / (c**3)
    denominator = 1 / (np.exp((h * (x * 3e10)) / (Kb * T)) - 1)
    spectrum = numerator * denominator
    return spectrum / np.sum(spectrum)

filenameCMB = 'C:/Users/lando/PycharmProjects/PHYS-270L/BlackbodyRad-Analysis/firas_monopole_spec_v1.txt'
freqCMB, countsCMB = np.genfromtxt(filenameCMB, usecols=(0, 1), skip_header=18, unpack=True)
countsCMB /= np.sum(countsCMB)  # Normalize counts

TCMB, TCMBunc = fit_and_extract(BlackbodyFreq, freqCMB, countsCMB, p0=3)

print(f"The temperature of empty space is {TCMB:.2f} ± {TCMBunc:.2f} Kelvin")

plt.figure(figsize=(7, 5), dpi=300)
plt.plot(freqCMB, countsCMB, linestyle="solid", color="gold", label="CMB Measured Spectrum")
plt.plot(freqCMB, BlackbodyFreq(freqCMB, TCMB), linestyle="dashed", color="purple", label="CMB Fit Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Intensity (Counts)")
plt.legend()
plt.title("CMB Spectrum")
plt.show()



