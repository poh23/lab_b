import numpy as np
from tools.extract_max_peak import extract_max_peak
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# glass sanity check figure
# graph 1 - H&B vs number of glass sheets, graph 2 - H&B vs height w glass and wo glass, graph 3 - diagonal vs not diagonal glass

def get_data_for_glass_and_no_glass_measurments():
    num_of_mm = np.arange(0, 16, 2)
    ch1_avg_peak_voltages = []
    ch2_avg_peak_voltages = []
    ch1_avg_peak_voltages_old = []
    ch2_avg_peak_voltages_old = []
    ch1_avg_peak_voltages_no_glass = []
    ch2_avg_peak_voltages_no_glass = []
    for mm in num_of_mm:
        file_path = f'../data/high/Harhava_material2_1sheet_{mm}_space_high.csv'
        ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
        ch1_avg_peak_voltages.append(ch1_avg_peak_voltage)
        ch2_avg_peak_voltages.append(ch2_avg_peak_voltage)

        file_path_old = f"../data/high/Harhava_material2_1sheet_{mm}_space_high_old.csv"
        ch1_avg_peak_voltage_old, ch2_avg_peak_voltage_old = extract_max_peak(file_path_old)
        ch1_avg_peak_voltages_old.append(ch1_avg_peak_voltage_old)
        ch2_avg_peak_voltages_old.append(ch2_avg_peak_voltage_old)

        file_path_no_glass = f'../data/high/Harhava_material2_1sheet_air_{mm}mm.csv'
        ch1_avg_peak_voltage_no_glass, ch2_avg_peak_voltage_no_glass = extract_max_peak(file_path_no_glass)
        if mm == 12:
            ch2_avg_peak_voltage_no_glass = 1.8
        if mm == 14:
            ch2_avg_peak_voltage_no_glass = 1.8
        ch1_avg_peak_voltages_no_glass.append(ch1_avg_peak_voltage_no_glass)
        ch2_avg_peak_voltages_no_glass.append(ch2_avg_peak_voltage_no_glass)

    return num_of_mm, ch1_avg_peak_voltages, ch2_avg_peak_voltages, ch1_avg_peak_voltages_no_glass, ch2_avg_peak_voltages_no_glass, ch1_avg_peak_voltages_old, ch2_avg_peak_voltages_old

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
    ch2_avg_peak_voltages_old = []
    ch1_avg_peak_voltages_diag = []
    ch2_avg_peak_voltages_diag = []
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.ravel()

    num_of_mm, ch1_avg_peak_voltages_1, ch2_avg_peak_voltages_1, ch1_avg_peak_voltages_no_glass, ch2_avg_peak_voltages_no_glass, ch1_air_peak_voltages_old, ch2_air_peak_voltages_old = get_data_for_glass_and_no_glass_measurments()
    num_of_glass, ch1_voltages_only_glass, ch2_voltages_only_glass = create_list_lonely_glass_sheets()
    num_of_sheets, ch1_avg_peak_voltages, ch2_avg_peak_voltages, ch1_avg_peak_voltages_diag, ch2_avg_peak_voltages_diag, ch1_voltages_old, ch2_voltages_old = get_data_for_diagonal_and_not_diagonal_glass()

    # Set universal styling options
    font = {'family': 'serif', 'size': 14}

    # Graph 1
    axes[0].errorbar(num_of_glass, ch1_voltages_only_glass, color='b', markersize=5, fmt='o',
                     yerr=0.4, elinewidth=1, capsize=3)
    axes[0].set_ylabel(r'H (a.u.)', fontdict=font)
    axes[0].set_xlabel('Num of Glass Sheets', fontdict=font)
    axes[3].errorbar(num_of_glass, ch2_voltages_only_glass, color='g', markersize=5, fmt='o',
                     yerr=0.3, elinewidth=1, capsize=3)
    axes[3].set_ylabel(r'$\Phi_B$ (a.u.)', fontdict=font)
    axes[3].set_xlabel('Num of Glass Sheets', fontdict=font)
    axes[3].set_ylim(0, 1)

    # Graph 2
    axes[1].errorbar(num_of_mm, ch1_avg_peak_voltages_1, color='b', markersize=5, fmt='o', label='With Glass',
                     yerr=0.4, elinewidth=1, capsize=3)
    # axes[1].scatter(num_of_mm, ch1_air_peak_voltages_old, color='g', s=30, marker='o', label='With Glass Old')
    axes[1].errorbar(num_of_mm, ch1_avg_peak_voltages_no_glass, color='r', markersize=5, fmt='o', label='Without Glass', xerr=1, yerr=0.5, elinewidth=1, capsize=3)
    axes[1].set_ylabel(r'H (a.u.)', fontdict=font)
    axes[1].set_xlabel('Height (mm)', fontdict=font)

    axes[4].errorbar(num_of_mm, ch2_avg_peak_voltages_1, color='b', markersize=5, fmt='o', label='With Glass',
                    yerr=0.3, elinewidth=1, capsize=3)
    # axes[4].scatter(num_of_mm, ch2_air_peak_voltages_old, color='b', s=30, marker='o', label='With Glass Old')
    axes[4].errorbar(num_of_mm, ch2_avg_peak_voltages_no_glass, color='r', markersize=5, fmt='o', label='Without Glass', xerr=1, yerr=0.1, elinewidth=1, capsize=3)
    axes[4].set_ylabel(r'$\Phi_B$ (a.u.)', fontdict=font)
    axes[4].set_xlabel('Height (mm)', fontdict=font)

    # Graph 3
    axes[2].errorbar(num_of_sheets, ch1_avg_peak_voltages, color='b', markersize=5, fmt='o', label='Vertical',
                     yerr=0.4, elinewidth=1, capsize=3)
    # axes[5].scatter(num_of_sheets, ch2_voltages_old, color='b', s=30, marker='o', label='Not Diagonal Old')
    axes[2].errorbar(num_of_sheets, ch1_avg_peak_voltages_diag, color='r', markersize=5, fmt='o', label='Horizontal',
                     yerr=0.4, elinewidth=1, capsize=3)
    axes[2].set_ylabel(r'H (a.u.)', fontdict=font)
    axes[2].set_xlabel('Num of Glass Sheets', fontdict=font)

    axes[5].errorbar(num_of_sheets, ch2_avg_peak_voltages, color='b', markersize=5, fmt='o', label='Vertical',
                     yerr=0.3, elinewidth=1, capsize=3)
    # axes[5].scatter(num_of_sheets, ch2_voltages_old, color='b', s=30, marker='o', label='Not Diagonal Old')
    axes[5].errorbar(num_of_sheets, ch2_avg_peak_voltages_diag, color='r', markersize=5, fmt='o', label='Horizontal',
                    yerr=0.3, elinewidth=1, capsize=3)
    axes[5].set_ylabel(r'$\Phi_B$ (a.u.)', fontdict=font)
    axes[5].set_xlabel('Num of Glass Sheets', fontdict=font)


    # Add legends to the plots
    for idx, ax in enumerate(axes):
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.annotate(f"{chr(65 + idx)}",  # This gives 'a', 'b', 'c'...
                    xy=(-0.18, 0.98),  # Relative position (2% from left, 98% from bottom)
                    xycoords='axes fraction',
                    font=font,
                    fontsize=16,
                    fontweight='bold',
                    ha='left',
                    va='top')

    # Display the plots
    plt.tight_layout(w_pad=2.0, h_pad=1.0)
    plt.savefig('../graphs/glass_sanity_check.png')

create_glass_sanity_check_graphs()

