import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')

# todo: one section for single magnetic sheet - H vs number of glass sheets, B vs number of glass sheets (or total cross section). maybe graph of permeability vs number of glass sheets,
# todo: one section for reg measerments - H vs number of total cross section, B vs number of total cross section. maybe graph of permeability vs number of total cross section. cross section only of magnetic material
# todo: one section for high measurements - H vs number of total cross section, B vs number of total cross section. maybe graph of permeability vs number of total cross section. cross section only of magnetic material and glass

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
def extract_max_peak(file_path):
    # Example usage
    data = load_csv_to_dataframe(file_path)

    # Splitting the original data into two sections for each channel
    channel_1 = data.iloc[:, :5]  # First 6 columns for Channel 1
    channel_2 = data.iloc[:, 6:11]  # Last 6 columns for Channel 2

    # Renaming columns for better readability
    channel_1.columns = ['metadata_name', 'metadata_value', 'metadata_units', 'T', 'V']
    channel_2.columns = ['metadata_name', 'metadata_value', 'metadata_units', 'T', 'V']

    channel_1 = channel_1[['T', 'V']]
    channel_2 = channel_2[['T', 'V']]

    # Extracting peaks from Channel 1
    ch1_peaks, _ = find_peaks(channel_1['V'], distance=10, prominence=10)

    # Extracting the peaks from the Channel 2
    ch2_peaks, _ = find_peaks(channel_2['V'], distance=5, prominence=5)

    print(f"Peak Voltage values ch1: {channel_1['V'].iloc[ch1_peaks].values}")
    print(f"Peak Voltage values ch2: {channel_2['V'].iloc[ch2_peaks].values}")

    # calculating average peak voltage for each channel
    ch1_avg_peak_voltage = channel_1['V'].max()
    ch2_avg_peak_voltage = channel_2['V'].max()

    return ch1_avg_peak_voltage, ch2_avg_peak_voltage

magnet_area = 12.46 # mm^2

def create_list_of_all_peak_voltages(sheet_num):
    num_of_bars = np.arange(1, 7)
    ch1_avg_peak_voltages = []
    ch2_avg_peak_voltages = []
    for bar in num_of_bars:
        file_path = f"../data/reg/Harhava_material2_{bar}sheet_{sheet_num}_space.csv"
        ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
        ch1_avg_peak_voltages.append(ch1_avg_peak_voltage)
        ch2_avg_peak_voltages.append(ch2_avg_peak_voltage)

    magnet_areas = num_of_bars * magnet_area
    return magnet_areas, ch1_avg_peak_voltages, ch2_avg_peak_voltages

def plot_avg_peak_voltages():
    # Create two plots in two separate figures, one for each channel
    fig1, axes = plt.subplots(1,2, figsize=(10, 5))
    ax1, ax2 = axes

    # Generate data (assuming function create_list_of_all_peak_voltages is predefined)
    magnet_areas_1, ch1_avg_peak_voltages_1, ch2_avg_peak_voltages_1 = create_list_of_all_peak_voltages(0)
    magnet_areas_2, ch1_avg_peak_voltages_2, ch2_avg_peak_voltages_2 = create_list_of_all_peak_voltages(2)

    # Set universal styling options
    font = {'family': 'serif', 'size': 14}

    # Channel 1 Scatter Plot
    ax1.scatter(magnet_areas_1, ch1_avg_peak_voltages_1, label='V1 0 sheets', color='b', s=50, marker='o')
    ax1.scatter(magnet_areas_2, ch1_avg_peak_voltages_2, label='V1 2 sheets', color='g', s=50, marker='o')
    ax1.set_ylabel(r'H (a.u.)', fontdict=font)  # Using LaTeX for proportionate sign
    ax1.set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Channel 2 Scatter Plot
    ax2.scatter(magnet_areas_1, ch2_avg_peak_voltages_1, label='V2 0 sheets', color='r', s=50, marker='s')
    ax2.scatter(magnet_areas_2, ch2_avg_peak_voltages_2, label='V2 2 sheets', color='y', s=50, marker='s')
    ax2.set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)
    ax2.set_ylabel(r'$\Phi_B$ (a.u.)', fontdict=font)  # LaTeX for proportionate sign and Phi_B
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # Add legends to the plots
    ax1.legend()
    ax2.legend()

    # Display the plots
    plt.show()

def create_list_of_all_peak_voltages_for_sheets(sheet_num, num_of_bars):
    num_of_glass_sheets= np.arange(0, sheet_num)
    ch1_avg_peak_voltages = []
    ch2_avg_peak_voltages = []
    for sheet_num in num_of_glass_sheets:
        file_path = f"../data/high/Harhava_material2_{num_of_bars}sheet_{2*sheet_num}_space_high.csv"
        ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
        ch1_avg_peak_voltages.append(ch1_avg_peak_voltage)

        ch2_avg_peak_voltages.append(ch2_avg_peak_voltage)

    bar_area = 30 * 0.48 # mm^2
    glass_area = 30 * 1 # mm^2
    if num_of_bars == 1:
        total_cross_section = num_of_glass_sheets * 2 * glass_area + bar_area
    else:
        total_cross_section = num_of_glass_sheets * 2 * glass_area * (num_of_bars-1) + num_of_bars * bar_area

    return num_of_glass_sheets * 2, ch1_avg_peak_voltages, ch2_avg_peak_voltages


def plot_avg_bars_and_sheets_peaks():
    # Create two plots in two separate figures, one for each channel

    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
    axes1 = axes1.ravel()
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    axes2 = axes2.ravel()

    one_sheet_v1, one_sheet_v2  = extract_max_peak("../data/reg/Harhava_material2_1sheet_0_space.csv")
    # Generate data (assuming function create_list_of_all_peak_voltages is predefined)
    total_areas_1, ch1_avg_peak_voltages_1, ch2_avg_peak_voltages_1 = create_list_of_all_peak_voltages_for_sheets(8, 1)
    total_areas_2, ch1_avg_peak_voltages_2, ch2_avg_peak_voltages_2 = create_list_of_all_peak_voltages_for_sheets(8, 2)
    # total_areas_3, ch1_avg_peak_voltages_3, ch2_avg_peak_voltages_3 = create_list_of_all_peak_voltages_for_sheets(8, 3)


    # Set universal styling options
    font = {'family': 'serif', 'size': 14}

    # Channel 1 Scatter Plot
    axes1[0].scatter(total_areas_1, ch1_avg_peak_voltages_1, label='1 magnetic bar', color='b', s=50, marker='o')
    axes2[0].scatter(total_areas_2, ch1_avg_peak_voltages_2, label='2 magnetic bars', color='g', s=50, marker='o')
    axes2[0].scatter([0], one_sheet_v1, label='1 magnetic bars no glass', color='r', s=50, marker='o')
    axes1[0].set_ylabel(r'H (a.u.)', fontdict=font)
    axes2[0].set_ylabel(r'H (a.u.)', fontdict=font)
    axes1[0].set_xlabel('Total Cross section (mm$^2$)', fontdict=font)
    axes2[0].set_xlabel('Total Cross section (mm$^2$)', fontdict=font)
    axes1[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes2[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes1[0].tick_params(axis='both', which='major', labelsize=12)
    axes2[0].tick_params(axis='both', which='major', labelsize=12)

    # Channel 2 Scatter Plot
    axes1[1].scatter(total_areas_1, ch2_avg_peak_voltages_1, label='1 bar', color='b', s=50, marker='s')
    axes2[1].scatter(total_areas_2, ch2_avg_peak_voltages_2, label='V2 2 bars', color='g', s=50, marker='s')
    axes2[1].scatter([0], one_sheet_v2, label='1 magnetic bars no glass', color='r', s=50, marker='s')
    axes1[1].set_ylabel(r'$Phi_B$ (a.u.)', fontdict=font)
    axes2[1].set_ylabel(r'$Phi_B$ (a.u.)', fontdict=font)
    axes1[1].set_xlabel('Total Cross section (mm$^2$)', fontdict=font)
    axes2[1].set_xlabel('Total Cross section (mm$^2$)', fontdict=font)
    axes1[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes2[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes1[1].tick_params(axis='both', which='major', labelsize=12)
    axes2[1].tick_params(axis='both', which='major', labelsize=12)

    # Add legends to the plots
    for ax in axes1:
        ax.legend()
    for ax in axes2:
        ax.legend()

    # Display the plots
    plt.show()

def extract_data_from_files(file_path):
    # Example usage
    data = load_csv_to_dataframe(file_path)

    # Splitting the original data into two sections for each channel
    channel_1 = data.iloc[:, :5]  # First 6 columns for Channel 1
    channel_2 = data.iloc[:, 6:11]  # Last 6 columns for Channel 2

    # Renaming columns for better readability
    channel_1.columns = ['metadata_name', 'metadata_value', 'metadata_units', 'T', 'V']
    channel_2.columns = ['metadata_name', 'metadata_value', 'metadata_units', 'T', 'V']

    channel_1 = channel_1[['T', 'V']]
    channel_2 = channel_2[['T', 'V']]

    return channel_1, channel_2

def show_graphs_of_H_by_time():
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
    axes1 = axes1.ravel()

    file_path_4_space = f"../data/high/Harhava_material2_1sheet_4_space_high.csv"
    channel_1_4_space, channel_2_4_space = extract_data_from_files(file_path_4_space)

    file_path_6_space = f"../data/high/Harhava_material2_1sheet_6_space_high.csv"
    channel_1_6_space, channel_2_6_space = extract_data_from_files(file_path_6_space)

    axes1[0].plot(channel_1_4_space['T'], channel_1_4_space['V'], label='H 4 space', c='g')
    axes1[0].plot(channel_1_6_space['T'], channel_1_6_space['V'], label='H 6 space', c='orange', linestyle='--')
    axes1[1].plot(channel_2_4_space['T'], channel_2_4_space['V'], label='B 4 space', c='g')
    axes1[1].plot(channel_2_6_space['T'], channel_2_6_space['V'], label='B 6 space', c='orange', linestyle='--')

    for ax in axes1:
        ax.legend()

    plt.show()




# plot_avg_bars_and_sheets_peaks()
#show_graphs_of_H_by_time()
#plot_avg_peak_voltages()

plot_avg_peak_voltages()








