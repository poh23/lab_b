import numpy as np
from tools.extract_max_peak import extract_max_peak
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# glass sanity check figure
# graph 1 - H&B vs number of glass sheets 2 bars material 2 and material 5, graph 2 - H&B vs number of glass sheets 3 bars material 2

def create_list_voltages_for_different_space_nums_2_sheets(material_num):
    spaces = np.arange(0, 16, 2)
    ch1_avg_peak_voltages = []
    ch2_avg_peak_voltages = []
    for space in spaces:
        file_path = f"../data/high/Harhava_material{material_num}_2sheet_{space}_space_high.csv"
        ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
        ch1_avg_peak_voltages.append(ch1_avg_peak_voltage)
        ch2_avg_peak_voltages.append(ch2_avg_peak_voltage)

    return spaces, ch1_avg_peak_voltages, ch2_avg_peak_voltages

def create_list_voltages_for_different_space_nums_3_sheets():
    spaces = np.arange(0, 16, 2)
    ch1_avg_peak_voltages = []
    ch2_avg_peak_voltages = []
    for space in spaces:
        file_path = f"../data/high/Harhava_material2_3sheet_{space}_space_high.csv"
        ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
        ch1_avg_peak_voltages.append(ch1_avg_peak_voltage)
        ch2_avg_peak_voltages.append(ch2_avg_peak_voltage)

    return spaces, ch1_avg_peak_voltages, ch2_avg_peak_voltages

def get_max_of_one_sheet():
    file_path = f"../data/high/Harhava_material2_1sheet_0_space_high.csv"
    ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
    return ch1_avg_peak_voltage, ch2_avg_peak_voltage


def plot_2_3_sheet_graphs():
    # Create two plots in two separate figures, one for each channel
    fig, axes = plt.subplots(2,1, figsize=(10, 10))
    axes = axes.ravel()

    mat2_spaces, ch1_peak_voltages_mat2, ch2_peak_voltages_mat2 = create_list_voltages_for_different_space_nums_2_sheets(2)
    ch1_one_sheet, ch2_one_sheet = get_max_of_one_sheet()

    # Set universal styling options
    font = {'family': 'serif', 'size': 14}

    # graph 1 - H&B vs number of glass sheets of under one sheet material 2 and material 5
    axes[0].errorbar(mat2_spaces, ch1_peak_voltages_mat2, yerr=0.4, fmt='o', label='Material 2', markersize=5, capsize=3)
    axes[0].axhline(y=ch1_one_sheet, color='black', linestyle='--', label='Material 2 1 sheet')
    axes[0].set_ylabel('H (a.u.)', fontdict=font)

    axes[1].errorbar(mat2_spaces, ch2_peak_voltages_mat2, yerr=0.3, fmt='o', label='Material 2', markersize=5, capsize=3)
    axes[1].axhline(y=ch2_one_sheet, color='black', linestyle='--', label='Material 2 1 sheet')
    axes[1].set_ylabel('$\Phi_B$ (a.u.)', fontdict=font)


    # Add legends to the plots
    for ax in axes:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel('Height (mm)', fontdict=font)
        # ax.legend()

    # Display the plots
    plt.show()

plot_2_3_sheet_graphs()