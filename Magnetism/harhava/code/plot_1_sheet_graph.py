import numpy as np
from tools.extract_max_peak import extract_max_peak
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec

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
    # Create a figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes = axes.ravel()

    mat2_spaces, ch1_peak_voltages_mat2, ch2_peak_voltages_mat2 = create_list_voltages_for_different_space_nums(2)
    # ratio mat is H/ΦB
    ratio_mat2 = np.array(ch1_peak_voltages_mat2) / np.array(ch2_peak_voltages_mat2)

    # Set universal styling options
    font = {'family': 'serif', 'size': 14}

    # sec 1 - H&B vs height of one sheet material 2
    axes[0].errorbar(mat2_spaces, ch1_peak_voltages_mat2, markersize=5, fmt='o',
                     yerr=0.4, elinewidth=1, capsize=3)
    axes[0].set_ylabel('H (a.u.)', fontdict=font)
    axes[0].set_xlabel('Height (mm)', fontdict=font)

    axes[1].errorbar(mat2_spaces, ch2_peak_voltages_mat2, markersize=5, fmt='o',
                     yerr=0.3, elinewidth=1, capsize=3)
    axes[1].set_xlabel('Height (mm)', fontdict=font)
    axes[1].set_ylabel('$\Phi_B$ (a.u.)', fontdict=font)

    # y_err = np.sqrt((0.4 / np.array(ch2_peak_voltages_mat2))**2 + (0.3 * np.array(ch1_peak_voltages_mat2) / np.array(ch2_peak_voltages_mat2)**2)**2)

    # Graph 2 - Ratio vs. number of glass sheets for Material 2 and Material 5
    axes[2].errorbar(mat2_spaces, 1/ratio_mat2, markersize=5, fmt='o',
                     yerr=0, elinewidth=1, capsize=3)
    # Calculate linear fit
    slope, intercept = np.polyfit(mat2_spaces, 1 / ratio_mat2, 1)
    fit_line = slope * mat2_spaces + intercept

    print(f'Linear fit: y = {slope:.3f}x + {intercept:.3f}')

    # Plot linear fit
    axes[2].plot(mat2_spaces, fit_line, color='orange', label='Linear fit')

    axes[2].set_xlabel('Height (mm)', fontdict=font)
    axes[2].set_ylabel(r'$\frac{H}{\Phi_B}$ (a.u.)', fontdict=font)

    # Add legends to the plots
    for ax in axes:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Display the plots
    plt.tight_layout()
    plt.show()

plot_1_sheet_graphs()
