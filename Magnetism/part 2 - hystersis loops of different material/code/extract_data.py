import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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
def extract_data(file_path):
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


def smooth_data(data, window_size=11, polyorder=2):
    """
    Smooth the data using the Savitzky-Golay filter.

    Parameters:
    data (array-like): Input data to smooth.
    window_size (int): Window size of the filter (must be odd).
    polyorder (int): Polynomial order for fitting.

    Returns:
    array-like: Smoothed data.
    """
    return savgol_filter(data, window_size, polyorder)



def create_list_of_all_loops():
    num_of_materials = np.arange(1, 5)
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    for material in num_of_materials:
        file_path = f"../data/2.2_material{material}.csv"
        ch1_voltage, ch2_voltage = extract_data(file_path)
        ax1.scatter(ch1_voltage, ch2_voltage, label=f"Material {material}", s=10)

    font = {'family': 'serif', 'size': 18}
    plt.xlabel("H (a.u.)", fontdict=font)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=1.5)  # Bold horizontal line at y=0
    plt.axvline(0, color='black', linewidth=1.5)  # Bold vertical line at x=0
    plt.ylabel("B (a.u.)", fontdict=font)
    plt.legend(prop={'family': 'serif', 'size': 14})
    plt.show()

# create_list_of_all_loops()

def calculate_loop_area_with_uncertainty(h_values, b_values, material,h_uncertainties, b_uncertainties):

    upper_indices = np.where(np.diff(b_values, prepend=b_values[0]) > 0)[0]
    lower_indices = np.where(np.diff(b_values, prepend=b_values[0]) < 0)[0]

    H_upper, B_upper = h_values[upper_indices], b_values[upper_indices]
    upper_positive_indices = np.where(B_upper >= 0)
    upper_negative_indices = np.where(B_upper < 0)
    upper_h_positive_values = H_upper[upper_positive_indices]
    upper_b_positive_values = B_upper[upper_positive_indices]
    upper_h_negative_values = H_upper[upper_negative_indices]
    upper_b_negative_values = np.abs(B_upper[upper_negative_indices])

    H_lower, B_lower = h_values[lower_indices], b_values[lower_indices]
    lower_negative_indices = np.where(B_lower < 0)
    lower_positive_indices = np.where(B_lower >= 0)
    lower_h_positive_values = H_lower[lower_positive_indices]
    lower_b_positive_values = B_lower[lower_positive_indices]
    lower_h_negative_values = H_lower[lower_negative_indices]
    lower_b_negative_values = np.abs(B_lower[lower_negative_indices])

    if material == 1:
        upper_h_positive_values = upper_h_positive_values[10:-5]
        upper_b_positive_values = upper_b_positive_values[10:-5]

    if material == 2:
        upper_h_positive_values = upper_h_positive_values[20:-2]
        upper_b_positive_values = upper_b_positive_values[20:-2]

    if material == 3:
        upper_h_positive_values = upper_h_positive_values[20:-15]
        upper_b_positive_values = upper_b_positive_values[20:-15]
        lower_h_negative_values = lower_h_negative_values[0:-5]
        lower_b_negative_values = lower_b_negative_values[0:-5]

    if material == 4:
        upper_h_positive_values = upper_h_positive_values[5:-5]
        upper_b_positive_values = upper_b_positive_values[5:-5]
        lower_h_positive_values = lower_h_positive_values[5:-5]
        lower_b_positive_values = lower_b_positive_values[5:-5]
        lower_h_negative_values = lower_h_negative_values[0:-40]
        lower_b_negative_values = lower_b_negative_values[0:-40]

    # Calculate area using trapezoidal integration
    positive_area = np.abs(np.trapz(lower_b_positive_values, lower_h_positive_values)) - np.abs(np.trapz(upper_b_positive_values, upper_h_positive_values))
    negative_area = np.abs(np.trapz(upper_b_negative_values, upper_h_negative_values)) - np.abs(np.trapz(lower_b_negative_values, lower_h_negative_values))
    area = positive_area + negative_area

    # Propagate uncertainties
    area_uncertainty_squared = 0

    for i in range(len(h_values) - 1):
        dH = h_values[i + 1] - h_values[i]
        avg_B = (b_values[i] + b_values[i + 1]) / 2

        # Error propagation for this segment
        dA_i_B = dH / 2  # Contribution from B uncertainty
        dA_i_H = avg_B / 2  # Contribution from H uncertainty

        delta_A_i = np.sqrt(
            (dA_i_B * b_uncertainties[i]) ** 2 +
            (dA_i_B * b_uncertainties[i + 1]) ** 2 +
            (dA_i_H * h_uncertainties[i]) ** 2 +
            (dA_i_H * h_uncertainties[i + 1]) ** 2
        )

        area_uncertainty_squared += delta_A_i ** 2

    area_uncertainty = np.sqrt(area_uncertainty_squared)
    return area, area_uncertainty


def calculate_width_closest_to_B_zero(H, B, H_uncertainty):
    # Split into upper and lower branches
    upper_indices = np.where(np.diff(B, prepend=B[0]) > 0)[0]
    lower_indices = np.where(np.diff(B, prepend=B[0]) < 0)[0]

    H_upper, B_upper = H[upper_indices], B[upper_indices]
    H_lower, B_lower = H[lower_indices], B[lower_indices]
    H_upper_uncertainty = H_uncertainty[upper_indices]
    H_lower_uncertainty = H_uncertainty[lower_indices]

    # Find the points closest to B = 0 for both branches
    upper_closest_idx = np.argmin(np.abs(B_upper))
    lower_closest_idx = np.argmin(np.abs(B_lower))

    H_upper_at_B0 = H_upper[upper_closest_idx]
    H_lower_at_B0 = H_lower[lower_closest_idx]

    # Uncertainties in H
    H_upper_unc_at_B0 = H_upper_uncertainty[upper_closest_idx]
    H_lower_unc_at_B0 = H_lower_uncertainty[lower_closest_idx]

    # Distance from B = 0 for the closest points
    B_upper_dist = np.abs(B_upper[upper_closest_idx])
    B_lower_dist = np.abs(B_lower[lower_closest_idx])

    # Add an additional uncertainty term proportional to |B| distance
    additional_uncertainty = 0.1  # Fractional uncertainty for extrapolation, adjust as needed
    extra_unc_upper = additional_uncertainty * B_upper_dist
    extra_unc_lower = additional_uncertainty * B_lower_dist

    # Calculate the width and total uncertainty
    width = abs(H_upper_at_B0 - H_lower_at_B0)
    uncertainty = np.sqrt(
        H_upper_unc_at_B0 ** 2 + H_lower_unc_at_B0 ** 2 + extra_unc_upper ** 2 + extra_unc_lower ** 2
    )

    return width, uncertainty


def generate_hystersis_graphs_and_calculate_data():
    num_of_materials = np.arange(1, 5)
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    for material in num_of_materials:
        file_path = f"../data/2.2_material{material}.csv"
        ch1_voltage, ch2_voltage = extract_data(file_path)
        ch1_smoothed_v = smooth_data(ch1_voltage['V'], window_size=41, polyorder=3)
        ch2_smoothed_v = smooth_data(ch2_voltage['V'], window_size=41, polyorder=3)
        ch1_smoothed_v_peaks, _ = find_peaks(ch1_smoothed_v, distance=10, prominence=10)
        ch2_smoothed_v_peaks, _ = find_peaks(ch2_smoothed_v, distance=50, prominence=10)

        ch1_smoothed_v_in_one_loop = ch1_smoothed_v[ch1_smoothed_v_peaks[0]:ch1_smoothed_v_peaks[1]]
        ch1_v_in_one_loop = ch1_voltage['V'][ch1_smoothed_v_peaks[0]:ch1_smoothed_v_peaks[1]].values
        ch2_smoothed_v_in_one_loop = ch2_smoothed_v[ch1_smoothed_v_peaks[0]:ch1_smoothed_v_peaks[1]]
        ch2_v_in_one_loop = ch2_voltage['V'][ch1_smoothed_v_peaks[0]:ch1_smoothed_v_peaks[1]].values

        ch1_smoothed_reg_diff = abs(ch1_smoothed_v_in_one_loop-ch1_v_in_one_loop)
        ch2_smoothed_reg_diff = abs(ch2_smoothed_v_in_one_loop-ch2_v_in_one_loop)

        area, area_uncertainty = calculate_loop_area_with_uncertainty(ch1_smoothed_v_in_one_loop, ch2_smoothed_v_in_one_loop, material, ch1_smoothed_reg_diff, ch2_smoothed_reg_diff)
        width, width_uncertainty = calculate_width_closest_to_B_zero(ch1_smoothed_v_in_one_loop, ch2_smoothed_v_in_one_loop, ch1_smoothed_reg_diff)
        ax1.scatter(ch1_v_in_one_loop, ch2_v_in_one_loop, s=10, label=f"Material {material}")
        print('Area:', area, 'Area Uncertainty:', area_uncertainty, 'Material:', material)
        print('Width:', width, 'Width Uncertainty:', width_uncertainty, 'Material:', material)

    font = {'family': 'serif', 'size': 18}
    ax1.set_xlabel("H (a.u.)", fontdict=font)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.axhline(0, color='black', linewidth=1.5)  # Bold horizontal line at y=0
    ax1.axvline(0, color='black', linewidth=1.5)  # Bold vertical line at x=0
    ax1.set_ylabel("B (a.u.)", fontdict=font)
    # ax1.legend(prop={'family': 'serif', 'size': 14})
    plt.show()

generate_hystersis_graphs_and_calculate_data()



