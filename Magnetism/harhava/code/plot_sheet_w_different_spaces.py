import matplotlib.pyplot as plt
import numpy as np
from tools.extract_max_peak import extract_max_peak
import matplotlib

matplotlib.use('TkAgg')

magnet_area_2 = 12.46 # mm^2
magnet_area_5 = 12.46 # mm^2

def create_list_of_all_peak_voltages_for_defferent_space_nums(magnet_area, material_num, num_spaces):
    num_of_bars = np.arange(1, 7)
    ch1_avg_peak_voltages = []
    ch2_avg_peak_voltages = []
    for bar in num_of_bars:
        file_path = f"../data/reg/Harhava_material{material_num}_{bar}sheet_{num_spaces}_space.csv"
        ch1_avg_peak_voltage, ch2_avg_peak_voltage = extract_max_peak(file_path)
        ch1_avg_peak_voltages.append(ch1_avg_peak_voltage)
        ch2_avg_peak_voltages.append(ch2_avg_peak_voltage)

    magnet_areas = num_of_bars * magnet_area
    return magnet_areas, ch1_avg_peak_voltages, ch2_avg_peak_voltages

def plot_sheet_w_different_spaces():
    # Create two plots in two separate figures, one for each channel
    fig, axes = plt.subplots(2,2, figsize=(10, 5))
    axes = axes.ravel()

    magnet_areas_mat2_0_spaces, ch1_peak_voltages_mat2_0_spaces, ch2_peak_voltages_mat2_0_spaces = create_list_of_all_peak_voltages_for_defferent_space_nums(magnet_area_2,2,0)
    magnet_areas_mat2_2_spaces, ch1_peak_voltages_mat2_2_spaces, ch2_peak_voltages_mat2_2_spaces = create_list_of_all_peak_voltages_for_defferent_space_nums(magnet_area_2,2,2)
    magnet_areas_mat2_4_spaces, ch1_peak_voltages_mat2_4_spaces, ch2_peak_voltages_mat2_4_spaces = create_list_of_all_peak_voltages_for_defferent_space_nums(magnet_area_2,2,4)

    magnet_areas_mat5_0_spaces, ch1_peak_voltages_mat5_0_spaces, ch2_peak_voltages_mat5_0_spaces = create_list_of_all_peak_voltages_for_defferent_space_nums(magnet_area_5,5,0)

    # Set universal styling options
    font = {'family': 'serif', 'size': 14}

    # H Scatter Plot material 2
    axes[0].scatter(magnet_areas_mat2_0_spaces, ch1_peak_voltages_mat2_0_spaces, label='0 sheets in between', color='b', s=50, marker='o')
    axes[0].scatter(magnet_areas_mat2_2_spaces, ch1_peak_voltages_mat2_2_spaces, label='2 sheets in between', color='g', s=50, marker='o')
    axes[0].scatter(magnet_areas_mat2_4_spaces, ch1_peak_voltages_mat2_4_spaces, label='4 sheets in between', color='r', s=50, marker='o')
    axes[0].set_ylabel(r'H (a.u.)', fontdict=font)
    axes[0].set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)

    # Phi_b Scatter Plot material 2
    axes[1].scatter(magnet_areas_mat2_0_spaces, ch2_peak_voltages_mat2_0_spaces, label='0 sheets in between', color='b', s=50, marker='o')
    axes[1].scatter(magnet_areas_mat2_2_spaces, ch2_peak_voltages_mat2_2_spaces, label='2 sheets in between', color='g', s=50, marker='o')
    axes[1].scatter(magnet_areas_mat2_4_spaces, ch2_peak_voltages_mat2_4_spaces, label='4 sheets in between', color='r', s=50, marker='o')
    axes[1].set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)
    axes[1].set_ylabel(r'$\Phi_B$ (a.u.)', fontdict=font)

    # material 5 subplots
    # H Scatter Plot
    axes[2].scatter(magnet_areas_mat5_0_spaces, ch1_peak_voltages_mat5_0_spaces, label='0 sheets in between', color='b', s=50, marker='o')
    axes[2].set_ylabel(r'H (a.u.)', fontdict=font)
    axes[2].set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)

    # B Scatter Plot
    axes[3].scatter(magnet_areas_mat5_0_spaces, ch2_peak_voltages_mat5_0_spaces, label='0 sheets in between', color='b', s=50, marker='o')
    axes[3].set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)
    axes[3].set_ylabel(r'$\Phi_B$ (a.u.)', fontdict=font)

    # Add legends to the plots
    for ax in axes:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()

    # Add group titles
    fig.text(0.5, 0.95, 'Material 2', ha='center', va='center', fontdict=font)
    fig.text(0.5, 0.5, 'Material 5', ha='center', va='center', fontdict=font)

    # Display the plots
    plt.show()

plot_sheet_w_different_spaces()
