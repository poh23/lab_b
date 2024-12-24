import matplotlib.pyplot as plt
import numpy as np
from tools.extract_max_peak import extract_max_peak
import matplotlib

matplotlib.use('TkAgg')

magnet_area_2 = 12.32 # mm^2 # width = 22 mm, height = 0.56 mm
magnet_area_5 = 10.25 # mm^2 # width = 25 mm, height = 0.41 mm

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
    fig1, axes1 = plt.subplots(2,1, figsize=(10, 10))
    axes1 = axes1.ravel()

    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 10))
    axes2 = axes2.ravel()

    magnet_areas_mat2_0_spaces, ch1_peak_voltages_mat2_0_spaces, ch2_peak_voltages_mat2_0_spaces = create_list_of_all_peak_voltages_for_defferent_space_nums(magnet_area_2,2,0)
    magnet_areas_mat2_2_spaces, ch1_peak_voltages_mat2_2_spaces, ch2_peak_voltages_mat2_2_spaces = create_list_of_all_peak_voltages_for_defferent_space_nums(magnet_area_2,2,2)
    magnet_areas_mat2_4_spaces, ch1_peak_voltages_mat2_4_spaces, ch2_peak_voltages_mat2_4_spaces = create_list_of_all_peak_voltages_for_defferent_space_nums(magnet_area_2,2,4)

    magnet_areas_mat5_0_spaces, ch1_peak_voltages_mat5_0_spaces, ch2_peak_voltages_mat5_0_spaces = create_list_of_all_peak_voltages_for_defferent_space_nums(magnet_area_5,5,0)

    # Set universal styling options
    font = {'family': 'serif', 'size': 14}

    # H Scatter Plot material 2
    axes1[0].errorbar(magnet_areas_mat2_0_spaces, ch1_peak_voltages_mat2_0_spaces, color='b', markersize=5, fmt='o',
                     yerr=0.4, elinewidth=1, capsize=3, label='0 mm in between')
    axes1[0].errorbar(magnet_areas_mat2_2_spaces, ch1_peak_voltages_mat2_2_spaces, color='g', markersize=5, fmt='o',
                      yerr=0.4, elinewidth=1, capsize=3, label='2 mm in between')
    axes1[0].errorbar(magnet_areas_mat2_4_spaces, ch1_peak_voltages_mat2_4_spaces, color='r', markersize=5, fmt='o',
                      yerr=0.4, elinewidth=1, capsize=3, label='4 mm in between')
    axes1[0].set_ylabel(r'H (a.u.)', fontdict=font)
    axes1[0].set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)

    # Phi_b Scatter Plot material 2
    axes1[1].errorbar(magnet_areas_mat2_0_spaces, ch2_peak_voltages_mat2_0_spaces, color='b', markersize=5, fmt='o',
                      yerr=0.3, elinewidth=1, capsize=3, label='0 mm in between')
    axes1[1].errorbar(magnet_areas_mat2_2_spaces, ch2_peak_voltages_mat2_2_spaces, color='g', markersize=5, fmt='o',
                      yerr=0.3, elinewidth=1, capsize=3, label='2 mm in between')
    axes1[1].errorbar(magnet_areas_mat2_4_spaces, ch2_peak_voltages_mat2_4_spaces, color='r', markersize=5, fmt='o',
                      yerr=0.3, elinewidth=1, capsize=3, label='4 mm in between')
    axes1[1].set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)
    axes1[1].set_ylabel(r'$\Phi_B$ (a.u.)', fontdict=font)

    # material 5 subplots
    # H Scatter Plot
    axes2[0].errorbar(magnet_areas_mat5_0_spaces, ch1_peak_voltages_mat5_0_spaces, color='b', markersize=5, fmt='o',
                      yerr=0.4, elinewidth=1, capsize=3, label='Material 5')
    axes2[0].errorbar(magnet_areas_mat2_0_spaces, ch1_peak_voltages_mat2_0_spaces, color='orange', markersize=5, fmt='o',
                      yerr=0.4, elinewidth=1, capsize=3, label='Material 2')
    axes2[0].set_ylabel(r'H (a.u.)', fontdict=font)
    axes2[0].set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)

    # B Scatter Plot
    axes2[1].errorbar(magnet_areas_mat5_0_spaces, ch2_peak_voltages_mat5_0_spaces, color='b', markersize=5, fmt='o',
                      yerr=0.3, elinewidth=1, capsize=3, label='Material 5')
    axes2[1].errorbar(magnet_areas_mat2_0_spaces, ch2_peak_voltages_mat2_0_spaces, color='orange', markersize=5, fmt='o',
                      yerr=0.3, elinewidth=1, capsize=3, label='Material 2')
    axes2[1].set_xlabel('Magnet Cross-Section (mm$^2$)', fontdict=font)
    axes2[1].set_ylabel(r'$\Phi_B$ (a.u.)', fontdict=font)

    # Add legends to the plots
    for idx, ax in enumerate(axes1):
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.annotate(f"{chr(65 + idx)}",  # This gives 'a', 'b', 'c'...
                    xy=(-0.07, 0.98),  # Relative position (2% from left, 98% from bottom)
                    xycoords='axes fraction',
                    font = font,
                    fontsize = 16,
                    fontweight='bold',
                    ha='left',
                    va='top')
        #ax.legend()


    for idx, ax in enumerate(axes2):
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.annotate(f"{chr(65 + idx)}",  # This gives 'a', 'b', 'c'...
                    xy=(-0.07, 0.98),  # Relative position (2% from left, 98% from bottom)
                    xycoords='axes fraction',
                    font=font,
                    fontsize=16,
                    fontweight='bold',
                    ha='left',
                    va='top')
        #ax.legend()

    # Display the plots
    fig1.tight_layout(pad=2.0)
    fig2.tight_layout(pad=2.0)
    plt.show()

plot_sheet_w_different_spaces()