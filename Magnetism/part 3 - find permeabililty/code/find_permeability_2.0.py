import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib as mb
from scipy import stats
from matplotlib.colors import Normalize
import matplotlib.cm as cm

mb.use('TkAgg')


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


def find_simultaneous_max(ch1_v, ch2_v):
    # Find the index where ch1 is at its maximum
    ch1_max_idx = np.argmax(ch1_v)
    ch2_max_idx = np.argmax(ch2_v)
    # Get the corresponding values from both channels
    return ch1_v[ch1_max_idx], ch2_v[ch2_max_idx]

def calc_max_of_parabola(a,b,c,a_var,b_var,c_var):
    x_max = -b / (2 * a)
    y_max = a * x_max ** 2 + b * x_max + c
    y_max_var = np.sqrt((b ** 2 / (2 * a ** 2) * a_var) ** 2 + (b * b_var / (2 * a)) ** 2 + c_var ** 2)
    return y_max, y_max_var

def find_max_point_in_loop(ch1_t_in_loop, ch1_v_in_loop, ch2_t_in_loop, ch2_v_in_loop, T):
    ch1_fit = np.polyfit(ch1_t_in_loop[T // 3:-T // 3], ch1_v_in_loop[T // 3:-T // 3], 2, cov=True)
    ch1_coefficients = ch1_fit[0]
    ch1_variance = np.sqrt(np.diag(ch1_fit[1]))
    ch1_max, ch1_max_var = calc_max_of_parabola(*ch1_coefficients, *ch1_variance)

    ch2_fit = np.polyfit(ch2_t_in_loop[T // 3:-T // 3], ch2_v_in_loop[T // 3:-T // 3], 2, cov=True)
    ch2_coefficients = ch2_fit[0]
    ch2_variance = np.sqrt(np.diag(ch2_fit[1]))
    ch2_max, ch2_max_var = calc_max_of_parabola(*ch2_coefficients, *ch2_variance)

    return ch1_max, ch2_max, ch1_max_var, ch2_max_var




def smooth_data(data, window_size=11, polyorder=2):

    return savgol_filter(data, window_size, polyorder)

def find_peaks_in_loop(ch1, ch2):
    ch1_smoothed_v = smooth_data(ch1['V'], window_size=101, polyorder=2)
    ch2_smoothed_v = smooth_data(ch2['V'], window_size=101, polyorder=2)
    ch1_smoothed_v_peaks, _ = find_peaks(ch1_smoothed_v, distance=20, prominence=10)

    T = ch1_smoothed_v_peaks[1] - ch1_smoothed_v_peaks[0]

    ch1_in_loop = ch1['V'].iloc[T // 2 + ch1_smoothed_v_peaks[0]:T // 2 + ch1_smoothed_v_peaks[1]].values
    ch1_t_in_loop = ch1['T'].iloc[ch1_smoothed_v_peaks[0]:ch1_smoothed_v_peaks[1]].values

    ch2_in_loop = ch2['V'].iloc[T // 2 + ch1_smoothed_v_peaks[0]:T // 2 + ch1_smoothed_v_peaks[1]].values
    ch2_t_in_loop = ch2['T'].iloc[T // 2 + ch1_smoothed_v_peaks[0]:T // 2 + ch1_smoothed_v_peaks[1]].values

    return ch1_in_loop, ch1_t_in_loop, ch2_in_loop, ch2_t_in_loop, T


# Function to generate permeability from simultaneous maximum points
def generate_permeability_from_max_points():
    font = {'family': 'serif', 'size': 18}
    num_of_materials = np.arange(1, 5)
    resistance = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.ravel()  # Flatten axes array for easier indexing

    for idx, material in enumerate(num_of_materials):
        max_H_values = []
        max_Hs_residuals = []
        max_B_values = []
        max_Bs_residuals = []
        resistances = []

        for res in resistance:
            file_path = f"../data/2.3_material{material}_{res}kohm.csv"
            ch1, ch2 = extract_data(file_path)
            ch1_in_loop, ch1_t_in_loop, ch2_in_loop, ch2_t_in_loop, T = find_peaks_in_loop(ch1, ch2)

            max_H, max_B, max_H_residuals, max_B_residuals = find_max_point_in_loop(ch1_t_in_loop, ch1_in_loop,
                                                                                    ch2_t_in_loop, ch2_in_loop, T)
            max_H_values.append(max_H)
            max_Hs_residuals.append(max_H_residuals)
            max_B_values.append(max_B)
            max_Bs_residuals.append(max_B_residuals)
            resistances.append(res)

        # Calculate permeability (μ = B / H)
        permeability = np.array(max_B_values) / np.array(max_H_values)
        permeability_error = np.sqrt((np.array(max_Bs_residuals) / np.array(max_H_values)) ** 2 + (
                    np.array(max_Hs_residuals) * np.array(max_B_values)/ np.array(max_H_values)**2) ** 2)

        # Plot permeability vs resistance with a color map
        scatter = axes[idx].scatter(
            max_H_values, permeability, c=resistances, cmap='viridis', s=50, label='Data Points', zorder=2
        )

        # Error bars
        axes[idx].errorbar(
            max_H_values,
            permeability,
            xerr=max_Hs_residuals,
            yerr=permeability_error,
            fmt='none',  # Suppress markers for error bars
            ecolor='black',  # Error bar color
            elinewidth=1,  # Line width of error bars
            capsize=3,  # Size of the error bar caps
            label='Error Bars',
            zorder=1
        )
        axes[idx].set_xlabel("H (a.u.)", fontdict=font)
        axes[idx].set_ylabel("$\mu$ (a.u.)", fontdict=font)
        axes[idx].set_title(f"Material {material}", fontdict=font)
        axes[idx].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[idx].set_xlim([0, 60])
        axes[idx].set_ylim([0.1, 0.3])
        axes[idx].tick_params(axis='both', which='major', labelsize=12)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[idx])
        cbar.set_label('Resistance (kΩ)', fontdict=font)


    plt.tight_layout(pad=10.0)
    plt.show()


# Call the function
# generate_permeability_from_max_points()

def create_graph_of_multiple_hystersis_loops():
    global scatter
    num_of_materials = np.arange(1, 5)

    font = {'family': 'serif', 'size': 18}
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.ravel()


    for material in num_of_materials:
        idx = material - 1
        resistance = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        max_Hs = []
        max_Hs_residuals = []
        max_Bs = []
        max_Bs_residuals = []
        norm = Normalize(vmin=min(resistance), vmax=max(resistance))
        cmap = cm.viridis  # Choose a colormap
        for res in resistance:
            file_path = f"../data/2.3_material{material}_{res}kohm.csv"
            ch1, ch2 = extract_data(file_path)
            ch1_in_loop, ch1_t_in_loop, ch2_in_loop, ch2_t_in_loop, T = find_peaks_in_loop(ch1, ch2)

            max_H, max_B, max_H_residuals, max_B_residuals = find_max_point_in_loop(ch1_t_in_loop, ch1_in_loop, ch2_t_in_loop, ch2_in_loop, T)
            max_Hs.append(max_H)
            max_Hs_residuals.append(max_H_residuals)
            max_Bs.append(max_B)
            max_Bs_residuals.append(max_B_residuals)
            color = cmap(norm(res))
            axes[idx].scatter(ch1_in_loop, ch2_in_loop, color=color, s=1)

        axes[idx].errorbar(max_Hs, max_Bs, xerr=max_Hs_residuals ,yerr=max_Bs_residuals, fmt='o', color='black', markersize=5, elinewidth=1, capsize=3)
        axes[idx].set_xlabel("H (a.u.)", fontdict=font)
        axes[idx].set_ylabel("B (a.u.)", fontdict=font)
        axes[idx].set_title(f"Material {material}", fontdict=font)
        axes[idx].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[idx].axhline(0, color='black', linewidth=1.5)
        axes[idx].axvline(0, color='black', linewidth=1.5)
        axes[idx].tick_params(axis='both', which='major', labelsize=12)
        axes[idx].set_xlim([-70, 70])
        axes[idx].set_ylim([-10, 10])

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[idx])
        cbar.set_label('Resistance (kΩ)', fontdict=font)

    plt.tight_layout(pad=10.0)
    plt.show()


create_graph_of_multiple_hystersis_loops()

def find_max_point_in_loop_w_diff_params(ch1_t_in_loop, ch1_v_in_loop, ch2_t_in_loop, ch2_v_in_loop, T):
    t1 = ch1_t_in_loop[T // 3:-T // 3]
    v1 = ch1_v_in_loop[T // 3:-T // 3]
    ch1_fit = np.polyfit(t1, v1, 2, full=True)
    ch1_coefficients = ch1_fit[0]
    ch1_fit_func = np.polyval(ch1_coefficients, t1)

    t2 = ch2_t_in_loop[T // 3:-T // 3]
    v2 = ch2_v_in_loop[T // 3:-T // 3]
    ch2_fit = np.polyfit(t2, v2, 2, full=True)
    ch2_coefficients = ch2_fit[0]
    ch2_fit_func = np.polyval(ch2_coefficients, t2)

    return t1, v1, ch1_fit_func, t2, v2, ch2_fit_func

def create_graphs_showing_the_fits():
    global scatter
    num_of_materials = np.arange(1, 5)

    font = {'family': 'serif', 'size': 18}

    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 16))
    axes1 = axes1.ravel()

    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 16))
    axes2 = axes2.ravel()

    for material in num_of_materials:
        idx = material - 1
        resistance = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        max_Hs = []
        max_Hs_residuals = []
        max_Bs = []
        max_Bs_residuals = []
        norm = Normalize(vmin=min(resistance), vmax=max(resistance))
        cmap = cm.viridis  # Choose a colormap
        for res in resistance:
            file_path = f"../data/2.3_material{material}_{res}kohm.csv"
            ch1, ch2 = extract_data(file_path)
            ch1_in_loop, ch1_t_in_loop, ch2_in_loop, ch2_t_in_loop, T = find_peaks_in_loop(ch1, ch2)

            t1, v1, ch1_fit_func, t2, v2, ch2_fit_func = find_max_point_in_loop_w_diff_params(ch1_t_in_loop, ch1_in_loop, ch2_t_in_loop, ch2_in_loop, T)
            color = cmap(norm(res))
            axes1[idx].scatter(t1, v1, color=color, s=1)
            axes2[idx].scatter(t2, v2, color=color, s=1)
            axes1[idx].plot(t1, ch1_fit_func, color='black')
            axes2[idx].plot(t2, ch2_fit_func, color='black')

        axes1[idx].errorbar(max_Hs, max_Bs, xerr=max_Hs_residuals ,yerr=max_Bs_residuals, fmt='o', color='black', markersize=5, elinewidth=1, capsize=3)
        axes1[idx].set_xlabel("H (a.u.)", fontdict=font)
        axes1[idx].set_ylabel("B (a.u.)", fontdict=font)
        axes1[idx].set_title(f"Material {material}", fontdict=font)
        axes1[idx].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes1[idx].axhline(0, color='black', linewidth=1.5)
        axes1[idx].axvline(0, color='black', linewidth=1.5)
        axes1[idx].legend(prop={'family': 'serif', 'size': 14})
        axes1[idx].tick_params(axis='both', which='major', labelsize=12)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes1[idx])
        cbar.set_label('Resistance (kΩ)')

        axes2[idx].errorbar(max_Hs, max_Bs, xerr=max_Hs_residuals, yerr=max_Bs_residuals, fmt='o', color='black',
                            markersize=5, elinewidth=1, capsize=3)
        axes2[idx].set_xlabel("H (a.u.)", fontdict=font)
        axes2[idx].set_ylabel("B (a.u.)", fontdict=font)
        axes2[idx].set_title(f"Material {material}", fontdict=font)
        axes2[idx].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes2[idx].axhline(0, color='black', linewidth=1.5)
        axes2[idx].axvline(0, color='black', linewidth=1.5)
        axes2[idx].legend(prop={'family': 'serif', 'size': 14})
        axes2[idx].tick_params(axis='both', which='major', labelsize=12)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes2[idx])
        cbar.set_label('Resistance (kΩ)')

        fig1.suptitle('Fits of H max points', fontdict=font)
        fig2.suptitle('Fits of B max points', fontdict=font)

    plt.tight_layout(pad=10.0)
    plt.show()

# create_graphs_showing_the_fits()