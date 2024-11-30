# Python_Day6
#Activity: Calculating the Frequency of an LC Circuit
import math

def lc_frequency(inductance, capacitance):
    frequency = 1 / (2 * math.pi * math.sqrt(inductance * capacitance))
    return frequency

# Example usage:
L = 0.001  # Inductance in Henrys
C = 0.000001  # Capacitance in Farads
print(f"The frequency of the LC circuit is {lc_frequency(L, C):.2f} Hz")

#Data Acquisition
import random
import time

def acquire_data(duration, frequency):
    """
    Simulates data acquisition from a sensor over a specified duration and frequency.

    Parameters:
    duration (int): Duration of data acquisition in seconds.
    frequency (int): Frequency of data acquisition in Hz.

    Returns:
    list: Collected data points.
    """
    data = []
    num_samples = duration * frequency
    interval = 1 / frequency
    
    for _ in range(num_samples):
        # Simulate reading a data point from a sensor
        data_point = random.uniform(0, 100)  # Replace with actual data reading code
        data.append(data_point)
        
        # Wait for the next sample
        time.sleep(interval)
    
    return data

# Example usage:
duration = 10  # seconds
frequency = 1  # Hz (1 sample per second)
collected_data = acquire_data(duration, frequency)
print(f"Collected Data: {collected_data}")

#Signal Processing
import numpy as np
import matplotlib.pyplot as plt

def moving_average_filter(signal, window_size):
    """
    Applies a moving average filter to the input signal.

    Parameters:
    signal (list or numpy array): The input signal.
    window_size (int): The size of the moving window.

    Returns:
    numpy array: The filtered signal.
    """
    return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

# Example usage:
# Generate a noisy signal
np.random.seed(42)
time = np.linspace(0, 10, 1000)
signal = np.sin(time) + np.random.normal(0, 0.5, len(time))

# Apply the moving average filter
window_size = 10
filtered_signal = moving_average_filter(signal, window_size)

# Plot the original and filtered signals
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Original Signal')
plt.plot(time[:len(filtered_signal)], filtered_signal, label='Filtered Signal', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal Processing: Moving Average Filter')
plt.legend()
plt.show()

#Fault Detection
import numpy as np
import matplotlib.pyplot as plt

def generate_signal(duration, frequency):
    """
    Generates a noisy signal for testing.
    
    Parameters:
    duration (int): Duration in seconds.
    frequency (int): Frequency of the signal in Hz.
    
    Returns:
    numpy array: Generated signal.
    """
    np.random.seed(42)
    time = np.linspace(0, duration, duration * frequency)
    signal = np.sin(time) + np.random.normal(0, 0.5, len(time))
    return time, signal

def detect_faults(signal, threshold):
    """
    Detects faults in the signal based on a threshold.
    
    Parameters:
    signal (numpy array): Input signal.
    threshold (float): Threshold value for fault detection.
    
    Returns:
    list: Indices of detected faults.
    """
    faults = np.where(np.abs(signal) > threshold)[0]
    return faults

# Example usage:
duration = 10  # seconds
frequency = 100  # Hz
threshold = 1.0  # Fault threshold

# Generate a signal
time, signal = generate_signal(duration, frequency)

# Detect faults
faults = detect_faults(signal, threshold)

# Plot the signal and detected faults
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Signal')
plt.plot(time[faults], signal[faults], 'ro', label='Detected Faults')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Fault Detection in Signal')
plt.legend()
plt.show()

#Reporting
import matplotlib.pyplot as plt

def generate_report(signal, faults, filename="fault_report.txt"):
    """
    Generates a report of detected faults in the signal.

    Parameters:
    signal (numpy array): Input signal.
    faults (list): Indices of detected faults.
    filename (str): Name of the file to save the report.

    Returns:
    None
    """
    with open(filename, 'w') as file:
        file.write("Fault Detection Report\n")
        file.write("======================\n\n")
        file.write(f"Total number of data points: {len(signal)}\n")
        file.write(f"Number of faults detected: {len(faults)}\n\n")
        file.write("Detected Faults:\n")
        file.write("----------------\n")
        for idx in faults:
            file.write(f"Index: {idx}, Value: {signal[idx]:.2f}\n")
    
    print(f"Report generated and saved as {filename}")

# Example usage:
# Assuming `signal` is the input signal and `faults` are detected fault indices
duration = 10  # seconds
frequency = 100  # Hz
threshold = 1.0  # Fault threshold

# Generate a signal (simulated data)
time, signal = generate_signal(duration, frequency)

# Detect faults (using previous function)
faults = detect_faults(signal, threshold)

# Generate the report
generate_report(signal, faults)
