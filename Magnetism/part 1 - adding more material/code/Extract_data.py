import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')

# Function to load a CSV file into a DataFrame
def load_csv_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

# Function to extract metadata and data from a CSV file
def extract_avg_peak(file_path):
    # Example usage
    data = load_csv_to_dataframe(file_path)

    # Splitting the original data into two sections for each channel
    channel_1 = data.iloc[:, :5]  # First 6 columns for Channel 1
    channel_2 = data.iloc[:, 6:11]  # Last 6 columns for Channel 2

    # Renaming columns for better readability
    channel_1.columns = ['metadata_name', 'metadata_value', 'metadata_units', 'T', 'V']
    channel_2.columns = ['metadata_name', 'metadata_value', 'metadata_units', 'T', 'V']

    # Extracting metadata and data for each channel
    channel_1_metadata = channel_1[['metadata_name', 'metadata_value', 'metadata_units']].transpose()
    channel_1_metadata.columns = channel_1_metadata.iloc[0]
    channel_1_metadata = channel_1_metadata[1:]

    channel_2_metadata = channel_2[['metadata_name', 'metadata_value', 'metadata_units']].transpose()
    channel_2_metadata.columns = channel_2_metadata.iloc[0]
    channel_2_metadata = channel_2_metadata[1:]

    ch1_vertical_scale = float(channel_1_metadata["Vertical Scale"]["metadata_value"])  # V scale
    ch1_horizontal_scale = float(channel_1_metadata["Horizontal Scale"]["metadata_value"])  # T scale

    ch2_vertical_scale = float(channel_2_metadata["Vertical Scale"]["metadata_value"])  # V scale
    ch2_horizontal_scale = float(channel_2_metadata["Horizontal Scale"]["metadata_value"])  # T scale

    channel_1 = channel_1[['T', 'V']]
    channel_2 = channel_2[['T', 'V']]

    #
    # # Multiplying the voltage values by the vertical scale
    # channel_1['V'] = channel_1['V'] * ch1_vertical_scale
    # channel_2['V'] = channel_2['V'] * ch2_vertical_scale
    #
    # # Multiplying the time values by the horizontal scale
    # channel_1['T'] = channel_1['T'] * ch1_horizontal_scale
    # channel_2['T'] = channel_2['T'] * ch2_horizontal_scale

    # Extracting peaks from Channel 1
    ch1_peaks, _ = find_peaks(channel_1['V'], distance=10, prominence=10)

    # Extracting the peaks from the Channel 2
    ch2_peaks, _ = find_peaks(channel_2['V'], distance=10, prominence=10)

    print(f"Peak Voltage values ch1: {channel_1['V'].iloc[ch1_peaks].values}")
    print(f"Peak Voltage values ch2: {channel_2['V'].iloc[ch2_peaks].values}")

    # calculating average peak voltage for each channel
    ch1_avg_peak_voltage = channel_1['V'].iloc[ch1_peaks].mean()
    ch2_avg_peak_voltage = channel_2['V'].iloc[ch2_peaks].mean()

    return ch1_avg_peak_voltage, ch2_avg_peak_voltage

magnet_area = 12.46 # mm^2

def create_list_of_all_peak_voltages():
    num_of_bars = np.arange(1, 9)
    ch1_avg_peak_voltages = []
    ch2_avg_peak_voltages = []
    for bar in num_of_bars:
        file_path = f"../data/2.1_mes{bar}bar_2ch.csv"
        ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_avg_peak(file_path)
        ch1_avg_peak_voltages.append(ch1_avg_peak_voltage)
        ch2_avg_peak_voltages.append(ch2_avg_peak_voltage)

    magnet_areas = num_of_bars * magnet_area
    return magnet_areas, ch1_avg_peak_voltages, ch2_avg_peak_voltages

def plot_avg_peak_voltages():
    # Create two plots in two separate figures, one for each channel
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    # Generate data (assuming function create_list_of_all_peak_voltages is predefined)
    magnet_areas, ch1_avg_peak_voltages, ch2_avg_peak_voltages = create_list_of_all_peak_voltages()

    # Set universal styling options
    font = {'family': 'serif', 'size': 14}

    # Channel 1 Scatter Plot
    ax1.scatter(magnet_areas, ch1_avg_peak_voltages, label='V1', color='b', s=50, marker='o')
    ax1.set_ylabel(r'H (a.u.)', fontdict=font)  # Using LaTeX for proportionate sign
    ax1.set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Channel 2 Scatter Plot
    ax2.scatter(magnet_areas, ch2_avg_peak_voltages, label='V2', color='r', s=50, marker='s')
    ax2.set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)
    ax2.set_ylabel(r'$\Phi_B$ (a.u.)', fontdict=font)  # LaTeX for proportionate sign and Phi_B
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # Display the plots
    plt.show()


plot_avg_peak_voltages()







