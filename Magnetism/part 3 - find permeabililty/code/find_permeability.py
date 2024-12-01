import pandas as pd
import numpy as np
from scipy.signal import find_peaks
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


def find_simultaneous_max(ch1, ch2):
    # Find the index where ch1 is at its maximum
    ch1_max_idx = ch1['V'].idxmax()
    ch2_max_idx = ch2['V'].idxmax()
    # Get the corresponding values from both channels
    return ch1.loc[ch1_max_idx, 'V'], ch2.loc[ch2_max_idx, 'V']


# Function to generate permeability from simultaneous maximum points
def generate_permeability_from_max_points():
    font = {'family': 'serif', 'size': 18}
    num_of_materials = np.arange(1, 5)
    resistance = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.ravel()  # Flatten axes array for easier indexing

    for idx, material in enumerate(num_of_materials):
        max_H_values = []
        max_B_values = []
        resistances = []

        for res in resistance:
            file_path = f"../data/2.3_material{material}_{res}kohm.csv"
            ch1, ch2 = extract_data(file_path)

            # Find simultaneous maximum points
            max_H, max_B = find_simultaneous_max(ch1, ch2)
            max_H_values.append(max_H)
            max_B_values.append(max_B)
            resistances.append(res)

        # Calculate permeability (μ = B / H)
        permeability = np.array(max_B_values) / np.array(max_H_values)

        # Plot permeability vs resistance with a color map
        scatter = axes[idx].scatter(max_H_values, permeability, c=resistances, cmap='viridis', s=50)
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
            max_Bs = []
            norm = Normalize(vmin=min(resistance), vmax=max(resistance))
            cmap = cm.viridis  # Choose a colormap
            for res in resistance:
                file_path = f"../data/2.3_material{material}_{res}kohm.csv"
                ch1, ch2 = extract_data(file_path)
                max_H, max_B = find_simultaneous_max(ch1, ch2)
                max_Hs.append(max_H)
                max_Bs.append(max_B)
                color = cmap(norm(res))
                axes[idx].scatter(ch1['V'], ch2['V'], color=color, s=1)

            axes[idx].scatter(max_Hs, max_Bs, color='black', s=30, label='Peaks')
            axes[idx].set_xlabel("H (a.u.)", fontdict=font)
            axes[idx].set_ylabel("B (a.u.)", fontdict=font)
            axes[idx].set_title(f"Material {material}", fontdict=font)
            axes[idx].grid(True, which='both', linestyle='--', linewidth=0.5)
            axes[idx].axhline(0, color='black', linewidth=1.5)
            axes[idx].axvline(0, color='black', linewidth=1.5)
            axes[idx].legend(prop={'family': 'serif', 'size': 14})
            axes[idx].tick_params(axis='both', which='major', labelsize=12)
            axes[idx].set_xlim([-70, 70])
            axes[idx].set_ylim([-10, 10])

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes[idx])
            cbar.set_label('Resistance (kΩ)')

    plt.tight_layout(pad=10.0)
    plt.show()



create_graph_of_multiple_hystersis_loops()

def create_graphs_of_fits():
    global scatter
    num_of_materials = np.arange(1, 5)

    font = {'family': 'serif', 'size': 18}
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.ravel()

    for material in num_of_materials:
            idx = material - 1
            resistance = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            max_Hs = []
            max_Bs = []
            norm = Normalize(vmin=min(resistance), vmax=max(resistance))
            cmap = cm.viridis  # Choose a colormap
            for res in resistance:
                file_path = f"../data/2.3_material{material}_{res}kohm.csv"
                ch1, ch2 = extract_data(file_path)
                max_H, max_B = find_simultaneous_max(ch1, ch2)
                max_Hs.append(max_H)
                max_Bs.append(max_B)
                color = cmap(norm(res))
                axes[idx].scatter(ch1['V'], ch2['V'], color=color, s=1)

            axes[idx].scatter(max_Hs, max_Bs, color='black', s=30, label='Peaks')
            axes[idx].set_xlabel("H (a.u.)", fontdict=font)
            axes[idx].set_ylabel("B (a.u.)", fontdict=font)
            axes[idx].set_title(f"Material {material}", fontdict=font)
            axes[idx].grid(True, which='both', linestyle='--', linewidth=0.5)
            axes[idx].axhline(0, color='black', linewidth=1.5)
            axes[idx].axvline(0, color='black', linewidth=1.5)
            axes[idx].legend(prop={'family': 'serif', 'size': 14})
            axes[idx].tick_params(axis='both', which='major', labelsize=12)
            axes[idx].set_xlim([-70, 70])
            axes[idx].set_ylim([-10, 10])

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes[idx])
            cbar.set_label('Resistance (kΩ)')

    plt.tight_layout(pad=10.0)
    plt.show()
