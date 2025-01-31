import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.gridspec import GridSpec
from pyhelpers.store import save_fig


matplotlib.use('TkAgg')

font = {'family': 'serif', 'size': 18}
bar_cross_section = 1 # 1 cm^2

def extract_data():
    # Load the CSV file
    file_path = r'data/Harhava Data Pt. 2 - Sheet1 (5).csv'
    df = pd.read_csv(file_path)

    column_name = 'Name of file'

    # Extract material (e.g., carb_1, carb_2)
    df['Material'] = df[column_name].str.extract(r'^(carb_\d+)')

    # Extract measurement suffix (_1, _2, _3, or nothing)
    df['Measurement'] = df[column_name].str.extract(r'_(\d+)$')

    # Replace NaN in the Measurement column with '0' to indicate no suffix
    df['Measurement'] = df['Measurement'].fillna('0')

    return df

def plot_strain_isochromatics_graph(ax):
    df = extract_data()

    lengths = {'carb_1': '100 mm', 'carb_2': '140 mm'}

    # Plot the stress isochromatics
    for material in df['Material'].unique():
        df_subset = df[df['Material'] == material]
        df_subset = df_subset[df_subset['Measurement'] == '0']
        ax.errorbar(df_subset['Compression  (dL/L)']*100, df_subset['Num of Isochromes (center cross - section )'], fmt='o', xerr=df_subset['Compression error']*100,yerr=df_subset['error (center cross-section)'], label=lengths[material], markersize=3, capsize=1, elinewidth=1)

    ax.set_xlabel(r'$\varepsilon$ (%)', fontdict=font)
    ax.set_ylabel('N', fontdict=font)
    ax.axvline(x=1.0, color='orange', linestyle='--', linewidth=1.2)
    ax.axvline(x=1.2925, linestyle='--', linewidth=1.2)
    #ax.set_title('Stress vs Isochromes', fontdict=font)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(prop = {'family': 'serif', 'size': 12})
    plt.tight_layout()
    return ax


#plot_strain_isochromatics_graph()

def plot_ext_stress_vs_strain(ax):
    df = extract_data()

    lengths = {'carb_1': '100 mm', 'carb_2': '140 mm'}

    for material in df['Material'].unique():
        df_subset = df[df['Material'] == material]
        df_subset = df_subset[df_subset['Measurement'] == '0']
        ax.errorbar(df_subset['Compression  (dL/L)']*100, df_subset['Force (N)']/(100*bar_cross_section),
                    fmt='o', xerr=df_subset['Compression error']*100,
                    label=lengths[material], markersize=3, capsize=1, elinewidth=1)

    ax.set_xlabel(r'$\varepsilon$ (%)', fontdict=font)
    ax.axvline(x=1.0, color='orange', linestyle='--', linewidth=1.2)
    ax.axvline(x=1.2925, linestyle='--', linewidth=1.2)
    ax.set_ylabel(r'$\sigma_{ext}$ $[MPa]$', fontdict=font)
    #ax.set_title(r'Stress vs $\sigma_{ext}$', fontdict=font)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(prop = {'family': 'serif', 'size': 12})
    plt.tight_layout()
    return ax

#plot_ext_stress_vs_strain()


def plot_ext_stress_vs_strain_w_derivative(ax, material):
    df = extract_data()

    lengths = {'carb_1': '100 mm', 'carb_2': '140 mm'}
    color = {'carb_1': '#1f77b4', 'carb_2': 'orange'}
    yields = {'carb_1': 1.2925, 'carb_2': 1.0}

    df_subset = df[df['Material'] == material]
    df_subset = df_subset[df_subset['Measurement'] == '0']
    x = df_subset['Compression  (dL/L)'] * 100
    y = df_subset['Force (N)'] / (100*bar_cross_section)
    # Calculate numerical derivative
    dydx = np.gradient(y, x)
    ax.scatter(x, dydx, s=20, label=lengths[material], color=color[material])

    ax.axvline(x=yields[material], color=color[material], linestyle='--', linewidth=1.2)
    ax.set_xlabel(r'$\varepsilon$ (%)', fontdict=font)
    ax.set_ylabel(r'$\frac{d\sigma_{ext}}{d\varepsilon}$', fontdict=font)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5)
    #ax.legend(prop = {'family': 'serif', 'size': 12})

    plt.tight_layout()
    return ax


def plot_ext_stress_vs_isochromatics():
    df = extract_data()

    fig, ax = plt.subplots()
    for material in df['Material'].unique():
        for measurement in df['Measurement'].unique():
            df_subset = df[(df['Material'] == material) & (df['Measurement'] == measurement)]
            ax.errorbar(df_subset['Force (N)']/bar_cross_section, df_subset['Num of Isochromes (center cross - section )'],
                        fmt='o', yerr=df_subset['error (center cross-section)'],
                        label=f'{material} - {measurement}', markersize=3, capsize=1, elinewidth=1)

    ax.set_xlabel(r'$\sigma_{ext}$ $\frac{N}{cm^2}$', fontdict=font)
    ax.set_ylabel(r'Number of Isochromes', fontdict=font)
    #ax.set_title(r'Number of Isochromes vs $\sigma_{ext}$', fontdict=font)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.legend()
    plt.tight_layout()
    plt.savefig('harhava-graphs/stress_isochromatics.png', dpi=300)

#plot_ext_stress_vs_isochromatics()

def generate_figure():
    fig, axes = plt.subplots(2,2, figsize=(10, 8))
    axes = axes.ravel()

    plot_ext_stress_vs_strain(axes[0])
    plot_strain_isochromatics_graph(axes[1])
    plot_ext_stress_vs_strain_w_derivative(axes[2], 'carb_2')
    plot_ext_stress_vs_strain_w_derivative(axes[3], 'carb_1')

    # Annotate each subplot with a capital letter
    for i, ax in enumerate(axes):
        ax.annotate(
            f"{chr(65 + i)}",  # A, B, C, etc.
            xy=(-0.18, 0.95),  # Position near the top-left corner
            xycoords='axes fraction',
            fontsize=14,
            fontweight='bold',
            ha='left',
            va='top',)

    # Adjust layout and display
    plt.tight_layout()
    #plt.show()
    plt.savefig('harhava-graphs/stress_vs_strain_big_subplot.png', dpi=300)

#generate_figure()

import seaborn as sns

def plot_delta_N_vs_strain():
    df = extract_data()

    palette = sns.color_palette("deep")

    material = 'carb_2'
    measurements = [2, 3]  # Only use measurements 2 and 3

    # Create subplots for each measurement
    fig, axes = plt.subplots(1, len(measurements), figsize=(12.8, 4.8), sharex=True, sharey=True)  # Subplots for each measurement

    for i, measurement in enumerate(measurements):
        df_subset = df[(df['Material'] == material) & (df['Measurement'] == str(measurement))]

        # Calculate Delta N left and right with errors
        df_subset['Delta N left'] = df_subset['Num of Isochromes (center cross - section )'] - df_subset['Num of Isochromes left side (vertically)']
        df_subset['Delta N left error'] = np.sqrt(df_subset['error (center cross-section)']**2 + df_subset['error left (vertically)']**2)
        df_subset['Delta N right'] = df_subset['Num of Isochromes (center cross - section )'] - df_subset['Num of Isochromes right side (vertically)']
        df_subset['Delta N right error'] = np.sqrt(df_subset['error (center cross-section)']**2 + df_subset['error right (vertically)']**2)

        # Plot Delta N left
        axes[i].errorbar(
            df_subset['Compression  (dL/L)'] * 100,
            df_subset['Delta N left'],
            fmt='o',
            xerr=df_subset['Compression error'] * 100,
            yerr=df_subset['Delta N left error'],
            label=r'$\Delta N_{left}$',
            markersize=5,
            color=palette[9],
            capsize=3,
            elinewidth=1.5
        )

        # Plot Delta N right
        axes[i].errorbar(
            df_subset['Compression  (dL/L)'] * 100,
            df_subset['Delta N right'],
            fmt='s',
            xerr=df_subset['Compression error'] * 100,
            yerr=df_subset['Delta N right error'],
            label=r'$\Delta N_{right}$',
            markersize=5,
            color=palette[7],
            capsize=3,
            elinewidth=1.5
        )

        # Configure subplot
        axes[i].set_xlabel(r'$\varepsilon$ (%)', fontdict=font)
        axes[i].set_ylabel(r'$\Delta N$', fontdict=font)
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        axes[i].legend(prop={'family': 'serif', 'size': 14})

        # Add capital letters to each subplot
        #axes[i].text(-0.1, 1.1, chr(65 + i), transform=axes[i].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

        # Add a title for each measurement
        #axes[i].set_title(fr'Measurement {measurement}', fontdict=font)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'harhava-graphs/strain_delta-isochromatics-measurements-2-and-3.png', dpi=600)



#plot_delta_N_vs_strain()

def plot_isochromes():
    # Load the data
    file_path = 'data/iso_vs_y.csv'
    data = pd.read_csv(file_path)

    # Get unique values of F
    unique_comp_values = [0.17, 0.25, 0.33]

    # Initialize the plot
    plt.figure()

    palette = sns.color_palette("deep")

    # Plot data for each value of F with error bars
    for i, F_value in enumerate(unique_comp_values):
        subset = data[data['compression'] == F_value]
        plt.errorbar(
            subset['distance (cm)'],
            subset['num isochromes until distance'],
            xerr=subset['dist error'],
            yerr=subset['error'],
            label=f"{F_value} %",
            fmt='o',
            color=palette[i],
            capsize=3,
            elinewidth=1.5
        )

        fit = np.polyfit(subset['distance (cm)'], subset['num isochromes until distance'], 1, cov=True,
                         w=np.ones_like(subset['distance (cm)']))
        p = np.poly1d(fit[0])
        #intercept_var, slope_var = np.sqrt(np.diag(fit[1]))

        plt.plot(
            np.sort(subset['distance (cm)']),
            p(np.sort(subset['distance (cm)'])),
            "--",
            color=palette[i],
            linewidth=1.5
        )

    # Add labels, legend, and title
    plt.xlabel('y (cm)', fontdict=font)
    plt.ylabel('N', fontdict=font)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(title=r'$\sigma_{ext}$ Values', prop={'family': 'serif', 'size': 14}, title_fontsize=14)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'y_iso.png', dpi=600)

def plot_delta_N_vs_external_stress():
    df = extract_data()

    material = 'carb_2'
    for measurement in [2, 3]:
        df_subset = df[(df['Material'] == material)]
        df_subset = df_subset[df_subset['Measurement'] == str(measurement)]

        # Calculate Delta N left and right with errors
        df_subset['Delta N left'] = df_subset['Num of Isochromes (center cross - section )'] - df_subset['Num of Isochromes left side (vertically)']
        df_subset['Delta N left error'] = np.sqrt(df_subset['error (center cross-section)']**2 + df_subset['error left (vertically)']**2)
        df_subset['Delta N right'] = df_subset['Num of Isochromes (center cross - section )'] - df_subset['Num of Isochromes right side (vertically)']
        df_subset['Delta N right error'] = np.sqrt(df_subset['error (center cross-section)']**2 + df_subset['error right (vertically)']**2)

        # Create a figure for each measurement
        fig, ax = plt.subplots()

        # Plot Delta N left
        ax.errorbar(
            df_subset['Force (N)']/bar_cross_section,
            df_subset['Delta N left'],
            fmt='o',
            yerr=df_subset['Delta N left error'],
            label=f'Delta N left',
            markersize=3,
            capsize=1,
            elinewidth=1
        )

        # Plot Delta N right
        ax.errorbar(
            df_subset['Force (N)']/bar_cross_section,
            df_subset['Delta N right'],
            fmt='s',
            yerr=df_subset['Delta N right error'],
            label=f'Delta N right',
            markersize=3,
            capsize=1,
            elinewidth=1
        )

        # Configure the plot
        ax.set_xlabel(r'$\sigma_{ext}$ $\frac{N}{cm^2}$', fontdict=font)
        ax.set_ylabel(r'$\Delta N$', fontdict=font)
        ax.set_title(rf'{material} - Measurement {measurement}: $\Delta N$ vs $\sigma_e', fontdict=font)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()

        # Show the plot
        plt.tight_layout()
        plt.savefig(f'harhava-graphs/stress_delta-isochromatics-mes-{measurement}.png', dpi=300)

plot_delta_N_vs_strain()
plot_isochromes()