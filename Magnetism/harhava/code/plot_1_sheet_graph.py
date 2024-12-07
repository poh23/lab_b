import numpy as np
from tools.extract_max_peak import extract_max_peak
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# glass sanity check figure
# graph 1 - H&B vs number of glass sheets of under one sheet material 2 and material 5, graph 2 - H&B standing vs. not standing

def create_list_voltages_for_different_space_nums(material_num):
    spaces = np.arange(0, 24, 2)
    ch1_avg_peak_voltages = []
    ch2_avg_peak_voltages = []
    for space in spaces:
        file_path = f"../data/high/Harhava_material{material_num}_1sheet_{space}_space_high.csv"
        ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
        ch1_avg_peak_voltages.append(ch1_avg_peak_voltage)
        ch2_avg_peak_voltages.append(ch2_avg_peak_voltage)

    return spaces, ch1_avg_peak_voltages, ch2_avg_peak_voltages

def create_list_of_standing_not_standing_voltages():
    max_height = [0.52, 30]
    file_path = f"../data/high/Harhava_material2_1sheet_0_space_high.csv"
    ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
    file_path_standing = f"../data/high/Harhava_material2_1sheet_standing.csv"
    ch1_avg_peak_voltage_standing, ch2_avg_peak_voltage_standing = extract_max_peak(file_path_standing)

    ch1_avg_peak_voltages = [ch1_avg_peak_voltage, ch1_avg_peak_voltage_standing]
    ch2_avg_peak_voltages = [ch2_avg_peak_voltage, ch2_avg_peak_voltage_standing]

    return max_height, ch1_avg_peak_voltages, ch2_avg_peak_voltages


def plot_1_sheet_graphs():
    # Create two plots in two separate figures, one for each channel
    fig, axes = plt.subplots(2,2, figsize=(10, 5))
    axes = axes.ravel()

    mat2_spaces, ch1_peak_voltages_mat2, ch2_peak_voltages_mat2 = create_list_voltages_for_different_space_nums(2)
    mat5_spaces, ch1_peak_voltages_mat5, ch2_peak_voltages_mat5 = create_list_voltages_for_different_space_nums(5)

    max_height, ch1_avg_peak_voltages, ch2_avg_peak_voltages = create_list_of_standing_not_standing_voltages()

    # Set universal styling options
    font = {'family': 'serif', 'size': 14}

    # graph 1 - H&B vs number of glass sheets of under one sheet material 2 and material 5
    axes[0].scatter(mat2_spaces, ch1_peak_voltages_mat2, label='Material 2', s=50)
    axes[0].scatter(mat5_spaces, ch1_peak_voltages_mat5, label='Material 5', s=30)
    axes[0].set_xlabel('Number of glass sheets')
    axes[0].set_ylabel('H (a.u.)')

    axes[1].scatter(mat2_spaces, ch2_peak_voltages_mat2, label='Material 2', s=50)
    axes[1].scatter(mat5_spaces, ch2_peak_voltages_mat5, label='Material 5', s=30)
    axes[1].set_xlabel('Number of glass sheets')
    axes[1].set_ylabel('$\Phi_B$ (a.u.)')

    # graph 2 - H&B standing vs. not standing
    axes[2].scatter(max_height, ch1_avg_peak_voltages)
    axes[2].set_xlabel('Standing height (mm)')
    axes[2].set_ylabel('H (a.u.)')

    axes[3].scatter(max_height, ch2_avg_peak_voltages)
    axes[3].set_xlabel('Standing height (mm)')
    axes[3].set_ylabel('$\Phi_B$ (a.u.)')

    # Add legends to the plots
    for ax in axes:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()

    # Add group titles
    fig.text(0.5, 0.95, 'H & B vs. Number of sheets', ha='center', va='center', fontdict=font)
    fig.text(0.5, 0.5, 'Standing vs. not Standing', ha='center', va='center', fontdict=font)

    # Display the plots
    plt.show()

plot_1_sheet_graphs()