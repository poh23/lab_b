import numpy as np
from tools.extract_max_peak import extract_max_peak
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# glass sanity check figure
# graph 1 - H&B vs number of glass sheets, graph 2 - H&B vs height w glass and wo glass, graph 3 - diagonal vs not diagonal glass

def get_data_for_glass_and_no_glass_measurments():
    num_of_mm = np.arange(0, 14, 2)
    ch1_avg_peak_voltages = []
    ch2_avg_peak_voltages = []
    ch1_avg_peak_voltages_no_glass = []
    ch2_avg_peak_voltages_no_glass = []
    for mm in num_of_mm:
        file_path = f'../data/high/Harhava_material2_1sheet_{mm}_space_high.csv'
        ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
        ch1_avg_peak_voltages.append(ch1_avg_peak_voltage)
        ch2_avg_peak_voltages.append(ch2_avg_peak_voltage)

        file_path_no_glass = f'../data/high/Harhava_material2_1sheet_air_{mm}mm.csv'
        ch1_avg_peak_voltage_no_glass, ch2_avg_peak_voltage_no_glass = extract_max_peak(file_path_no_glass)
        ch1_avg_peak_voltages_no_glass.append(ch1_avg_peak_voltage_no_glass)
        ch2_avg_peak_voltages_no_glass.append(ch2_avg_peak_voltage_no_glass)

    return num_of_mm, ch1_avg_peak_voltages, ch2_avg_peak_voltages, ch1_avg_peak_voltages_no_glass, ch2_avg_peak_voltages_no_glass

def create_list_lonely_glass_sheets():
    num_of_glass_sheets = [0, 1, 3, 5, 8]
    ch1_avg_peak_voltages = []
    ch2_avg_peak_voltages = []
    for num_sheets in num_of_glass_sheets:
        file_path = f"../data/high/Harhava_0sheet_{num_sheets}glass.csv"
        ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
        ch1_avg_peak_voltages.append(ch1_avg_peak_voltage)
        ch2_avg_peak_voltages.append(ch2_avg_peak_voltage)

    return num_of_glass_sheets, ch1_avg_peak_voltages, ch2_avg_peak_voltages

def get_data_for_diagonal_and_not_diagonal_glass():
    num_of_sheets = np.arange(0, 14, 2)
    ch1_avg_peak_voltages = []
    ch2_avg_peak_voltages = []
    ch1_avg_peak_voltages_old = []
    ch1_avg_peak_voltages_diag = []
    ch2_avg_peak_voltages_diag = []
    ch2_avg_peak_voltages_old = []
    for sheet in num_of_sheets:
        file_path = f"../data/high/Harhava_material2_1sheet_{sheet}_space_high.csv"
        ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
        ch1_avg_peak_voltages.append(ch1_avg_peak_voltage)
        ch2_avg_peak_voltages.append(ch2_avg_peak_voltage)

        file_path_old = f"../data/high/Harhava_material2_1sheet_{sheet}_space_high_old.csv"
        ch1_avg_peak_voltage_old, ch2_avg_peak_voltage_old = extract_max_peak(file_path_old)
        ch1_avg_peak_voltages_old.append(ch1_avg_peak_voltage_old)
        ch2_avg_peak_voltages_old.append(ch2_avg_peak_voltage_old)

        file_path_no_glass = f'../data/high/Harhava_material2_1sheet_{sheet}_space_high_diag.csv'
        ch1_avg_peak_voltage_diag, ch2_avg_peak_voltage_diag = extract_max_peak(file_path_no_glass)
        ch1_avg_peak_voltages_diag.append(ch1_avg_peak_voltage_diag)
        ch2_avg_peak_voltages_diag.append(ch2_avg_peak_voltage_diag)

    return num_of_sheets, ch1_avg_peak_voltages, ch2_avg_peak_voltages, ch1_avg_peak_voltages_diag, ch2_avg_peak_voltages_diag, ch1_avg_peak_voltages_old, ch2_avg_peak_voltages_old

def create_glass_sanity_check_graphs():
    # Create two plots in two separate figures, one for each channel

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.ravel()

    num_of_mm, ch1_avg_peak_voltages_1, ch2_avg_peak_voltages_1, ch1_avg_peak_voltages_no_glass, ch2_avg_peak_voltages_no_glass = get_data_for_glass_and_no_glass_measurments()
    num_of_glass, ch1_voltages_only_glass, ch2_voltages_only_glass = create_list_lonely_glass_sheets()
    num_of_sheets, ch1_avg_peak_voltages, ch2_avg_peak_voltages, ch1_avg_peak_voltages_diag, ch2_avg_peak_voltages_diag = get_data_for_diagonal_and_not_diagonal_glass()

    # Set universal styling options
    font = {'family': 'serif', 'size': 14}

    # Graph 1
    axes[0].scatter(num_of_glass, ch1_voltages_only_glass, color='b', s=50, marker='o')
    axes[0].set_ylabel(r'H (a.u.)', fontdict=font)
    axes[0].set_xlabel('Num of Glass Sheets', fontdict=font)
    axes[1].scatter(num_of_glass, ch2_voltages_only_glass, color='g', s=50, marker='o')
    axes[1].set_ylabel(r'$Phi_B$ (a.u.)', fontdict=font)
    axes[1].set_xlabel('Num of Glass Sheets', fontdict=font)

    # Graph 2
    axes[2].scatter(num_of_mm, ch1_avg_peak_voltages_1, color='b', s=50, marker='o', label='With Glass')
    axes[2].scatter(num_of_mm, ch1_avg_peak_voltages_no_glass, color='r', s=30, marker='o', label='Without Glass')
    axes[2].set_ylabel(r'H (a.u.)', fontdict=font)
    axes[2].set_xlabel('Height (mm)', fontdict=font)

    axes[3].scatter(num_of_mm, ch2_avg_peak_voltages_1, color='g', s=50, marker='o', label='With Glass')
    axes[3].scatter(num_of_mm, ch2_avg_peak_voltages_no_glass, color='r', s=30, marker='o', label='Without Glass')
    axes[3].set_ylabel(r'$Phi_B$ (a.u.)', fontdict=font)
    axes[3].set_xlabel('Height (mm)', fontdict=font)

    # Graph 3
    axes[4].scatter(num_of_sheets, ch1_avg_peak_voltages, color='b', s=50, marker='o', label='Not Diagonal')
    axes[4].scatter(num_of_sheets, ch1_avg_peak_voltages_diag, color='r', s=30, marker='o', label='Diagonal')
    axes[4].set_ylabel(r'H (a.u.)', fontdict=font)
    axes[4].set_xlabel('Num of Glass Sheets', fontdict=font)

    axes[5].scatter(num_of_sheets, ch2_avg_peak_voltages, color='g', s=50, marker='o', label='Not Diagonal')
    axes[5].scatter(num_of_sheets, ch2_avg_peak_voltages_diag, color='r', s=30, marker='o', label='Diagonal')
    axes[5].set_ylabel(r'$Phi_B$ (a.u.)', fontdict=font)
    axes[5].set_xlabel('Num of Glass Sheets', fontdict=font)


    # Add legends to the plots
    for ax in axes:
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Display the plots
    plt.show()

create_glass_sanity_check_graphs()

