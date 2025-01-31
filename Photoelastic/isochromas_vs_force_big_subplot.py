import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.gridspec import GridSpec

matplotlib.use('TkAgg')


colors = {460: 'b', 525: 'g', 625: 'r'}
radii = {"disc1": 3.75, "disc2": 6.25, "disc3": 5}
font = {'family': 'serif', 'size': 16}
x_errors ={"disc1": 75, "disc2": 165, "disc3": 165, "disc4": 1.5}
lambda_fits = {'disc1': [2.83826 , -1.24502], 'disc2': [1.59588, -0.65006], 'disc3': [2.36006, -1.26963], 'disc4': [124.01003707,  -2.87474552]}
R_fits = {460: [0.00928439, 8.38719338], 525: [0.00797887, 6.22926252], 625: [0.00871047, 7.37956197]}

def extract_data():
    # Load the CSV file
    file_path = r'data/number_of_isochromes.csv'
    df = pd.read_csv(file_path)

    # Extract unique discs
    unique_discs = df['file name'].str.split('_').str[0].unique()
    unique_wavelengths = df['Lamda (wave length nm)'].unique()

    # Separate 'disc4' for individual plotting
    disc4_df = df[df['file name'].str.contains('disc4')]
    other_discs = [disc for disc in unique_discs if disc != 'disc4']

    return df, unique_discs, unique_wavelengths, disc4_df, other_discs


# plot 1 f/lamda plots for discs one through three
def plot_combined_R(ax):
    df, unique_discs, unique_wavelengths, disc4_df, other_discs = extract_data()

    markers = ['o', 's', 'D', '^', 'v', '<', '>']  # Different marker shapes for wavelengths
    colors = ['blue', 'green', 'orange']  # Colors for radii

    marker_map = {wavelength: markers[i % len(markers)] for i, wavelength in enumerate(unique_wavelengths)}
    color_map = {disc: colors[i % len(colors)] for i, disc in enumerate(other_discs)}

    for disc in other_discs:
        disc_df = df[df['file name'].str.contains(disc)]
        for wavelength in unique_wavelengths:
            wavelength_df = disc_df[disc_df['Lamda (wave length nm)'] == wavelength]
            ax.errorbar(
                wavelength_df['F/lambda'],
                wavelength_df['Num of Isochromes'],
                yerr=wavelength_df['error'],
                xerr=x_errors[disc] / wavelength,
                fmt=marker_map[wavelength],
                markersize=5,
                capsize=1,
                elinewidth=1,
                color=color_map[disc]
            )

        # Add linear fit for each radius
        fit = np.polyfit(disc_df['F/lambda'], disc_df['Num of Isochromes'], 1, cov=True)
        p = np.poly1d(fit[0])
        intercept_var, slope_var = np.sqrt(np.diag(fit[1]))
        ax.plot(
            disc_df['F/lambda'],
            p(disc_df['F/lambda']),
            "--",
            linewidth=0.9,
            label=f'R={radii[disc]} cm fit: {p[1]:.2f}x + {p[0]:.2f}',
            color=color_map[disc]
        )

        # Calculate R squared
        values = disc_df['Num of Isochromes'].values
        residuals = values - p(disc_df['F/lambda'])
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f'{disc} linear fit: {p[1]:.5f}x + {p[0]:.5f}')
        print(f'Slope error: {slope_var:.4f}')
        print(f'Intercept error: {intercept_var:.4f}')
        print(f'R squared: {r_squared:.4f}')

    # ax.set_title('Combined Plot for Different Radii and Wavelengths', fontdict=font)
    ax.set_xlabel(r'$\frac{F}{\lambda}$ ($\frac{N}{nm}$)', fontdict=font)
    ax.set_ylabel('N', fontdict=font)
    ax.legend(prop={'family': 'serif', 'size': 10})
    handles, labels = ax.get_legend_handles_labels()
    fit_handles = [handle for handle, label in zip(handles, labels) if 'fit' in label]
    fit_labels = [label for label in labels if 'fit' in label]
    ax.legend(fit_handles, fit_labels, prop={'family': 'serif', 'size': 10})
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    return ax

# plot 2 f/r plots for discs one through three
def plot_combined_wavelengths(ax):
    df, unique_discs, unique_wavelengths, disc4_df, other_discs = extract_data()
    all_disc_df = df[df['file name'].str.contains('|'.join(other_discs))]

    markers = ['o', 's', 'D', '^', 'v', '<', '>']  # Different marker shapes for radii

    marker_map = {disc: markers[i % len(markers)] for i, disc in enumerate(other_discs)}
    color_map = {460: 'b', 525: 'g', 625: 'r'}  # Colors for wavelengths

    for wavelength in unique_wavelengths:
        wavelength_df = all_disc_df[all_disc_df['Lamda (wave length nm)'] == wavelength]
        for disc in other_discs:
            disc_df = wavelength_df[wavelength_df['file name'].str.contains(disc)]
            ax.errorbar(
                disc_df['F/R'],
                disc_df['Num of Isochromes'],
                yerr=disc_df['error'],
                xerr=x_errors[disc] / radii[disc],
                fmt=marker_map[disc],
                markersize=5,
                capsize=1,
                elinewidth=1,
                color=color_map[wavelength]
            )

        # Add linear fit for each wavelength
        fit = np.polyfit(wavelength_df['F/R'], wavelength_df['Num of Isochromes'], 1, cov=True)
        p = np.poly1d(fit[0])
        intercept_var, slope_var = np.sqrt(np.diag(fit[1]))
        ax.plot(
            wavelength_df['F/R'],
            p(wavelength_df['F/R']),
            "--",
            linewidth=0.9,
            label=fr'$\lambda={wavelength}$ nm fit: {p[1]:.2f}x + {p[0]:.2f}',
            color=color_map[wavelength]
        )

        # Calculate R squared
        values = wavelength_df['Num of Isochromes'].values
        residuals = values - p(wavelength_df['F/R'])
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(fr'$\lambda$={wavelength} linear fit: {p[1]:.5f}x + {p[0]:.5f}')
        print(f'Slope error: {slope_var:.4f}')
        print(f'Intercept error: {intercept_var:.4f}')
        print(f'R squared: {r_squared:.4f}')

    # ax.set_title('Combined Plot for Different Wavelengths and Radii', fontdict=font)
    ax.set_xlabel(r'$\frac{F}{R}$ ($\frac{N}{cm}$)', fontdict=font)
    ax.set_ylabel('N', fontdict=font)
    handles, labels = ax.get_legend_handles_labels()
    fit_handles = [handle for handle, label in zip(handles, labels) if 'fit' in label]
    fit_labels = [label for label in labels if 'fit' in label]
    ax.legend(fit_handles, fit_labels, prop={'family': 'serif', 'size': 10})
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    return ax


# # Plot 3: Separate plots for disc4
def plot_Flamda_for_disc4(ax):
    df, unique_discs, unique_wavelengths, disc4_df, other_discs = extract_data()

    for wavelength in unique_wavelengths:
        wavelength_df = disc4_df[disc4_df['Lamda (wave length nm)'] == wavelength]
        ax.errorbar(wavelength_df['F/lambda'], wavelength_df['Num of Isochromes'], yerr=wavelength_df['error'], xerr=x_errors['disc4']/wavelength ,fmt='o', label=f'{wavelength} nm', markersize=5,
                         capsize=1, elinewidth=1, color=colors[wavelength])
        # Add linear fit
    fit = np.polyfit(disc4_df['F/lambda'], disc4_df['Num of Isochromes'], 1, cov=True)
    p = np.poly1d(fit[0])
    intercept_var, slope_var = np.sqrt(np.diag(fit[1]))
    ax.plot(disc4_df['F/lambda'], p(disc4_df['F/lambda']), "--", linewidth=0.9, color='black',
                 label=f'Linear fit: {p[1]:.2f}x + {p[0]:.2f}')
    print(f'disc4 linear fit: {p[1]:.5f}x + {p[0]:.5f}')

    # Calculate R squared
    values = disc4_df['Num of Isochromes'].values
    residuals = values - p(disc4_df['F/lambda'])
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f'Slope error: {slope_var:.4f}')
    print(f'Intercept error: {intercept_var:.4f}')
    print(f'R squared: {r_squared:.4f}')

    fits = {'disc4': fit[0]}
    print(fits)

    ax.set_xlabel(r'$\frac{F}{\lambda}$ ($\frac{N}{nm}$)', fontdict=font)
    ax.set_ylabel('N', fontdict=font)
    ax.legend(prop={'family': 'serif', 'size': 10})
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    return ax


# Plot 4: white discs
def plot_disc_2_white(ax):
    font = {'family': 'serif', 'size': 18}
    colors = ['blue', 'green', 'red']
    wave_lengths = {'blue': 460, 'green': 525, 'red': 625}
    # Load the CSV file
    file_path = r'data/white_isochromes_new.csv'
    df = pd.read_csv(file_path)

    disc_df = df[df['white files'].str.contains('disc2')]
    for color in colors:
        ax.errorbar(disc_df['F']/wave_lengths[color], disc_df[f'N {color} isochromes'], yerr=disc_df[f'{color}_error'], xerr=x_errors['disc2']/wave_lengths[color], fmt='o', label=f'{wave_lengths[color]} nm', markersize=5,
                         capsize=1, elinewidth=1, color=color)
    # Add linear fit
    p = np.poly1d(lambda_fits['disc2'])
    ax.plot(disc_df['F']/wave_lengths['blue'], p(disc_df['F']/wave_lengths['blue']), "--", linewidth=0.9, color='black')

    ax.set_xlabel(r'$\frac{F}{\lambda}$ ($\frac{N}{nm}$)', fontdict=font)
    ax.set_ylabel('N', fontdict=font)
    ax.legend(prop={'family': 'serif', 'size': 14})
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    return ax

def plot_normalized(ax):
    df, unique_discs, unique_wavelengths, disc4_df, other_discs = extract_data()
    all_disc_df = df[df['file name'].str.contains('|'.join(other_discs))]

    markers = ['o', 's', 'D', '^', 'v', '<', '>']  # Different marker shapes for radii

    marker_map = {disc: markers[i % len(markers)] for i, disc in enumerate(other_discs)}
    color_map = colors

    # Collect data for all measurements
    normalized_x = []
    normalized_y = []
    all_errors = []

    for wavelength in unique_wavelengths:
        wavelength_df = all_disc_df[all_disc_df['Lamda (wave length nm)'] == wavelength]
        for disc in other_discs:
            disc_df = wavelength_df[wavelength_df['file name'].str.contains(disc)]

            # Normalize F by R * wavelength
            norm_x = disc_df['F/R'] / wavelength
            normalized_x.extend(norm_x)
            normalized_y.extend(disc_df['Num of Isochromes'])
            all_errors.extend(disc_df['error'])

            ax.errorbar(
                norm_x,
                disc_df['Num of Isochromes'],
                yerr=disc_df['error'],
                fmt=marker_map[disc],
                markersize=5,
                capsize=1,
                elinewidth=1,
                color=color_map[wavelength],
                alpha=0.5
            )

    # Convert to numpy arrays for fitting
    normalized_x = np.array(normalized_x)
    normalized_y = np.array(normalized_y)

    # Perform a linear fit for all measurements
    fit = np.polyfit(normalized_x, normalized_y, 1, cov=True)
    p = np.poly1d(fit[0])
    intercept_var, slope_var = np.sqrt(np.diag(fit[1]))

    ax.plot(
        np.sort(normalized_x),
        p(np.sort(normalized_x)),
        "--",
        linewidth=1.5,
        label=f'Fit: {p[1]:.2f}x + {p[0]:.2f}',
        color='black'
    )

    # Calculate R squared
    residuals = normalized_y - p(normalized_x)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((normalized_y - np.mean(normalized_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f'Global linear fit: {p[1]:.5f}x + {p[0]:.5f}')
    print(f'Slope error: {slope_var:.4f}')
    print(f'Intercept error: {intercept_var:.4f}')
    print(f'R squared: {r_squared:.4f}')

    # ax.set_title('N vs. F / (R * Wavelength)', fontdict=font)
    ax.set_xlabel(r'$\frac{F}{R \cdot \lambda}$ ($\frac{N}{cm \cdot nm}$)', fontdict=font)
    ax.set_ylabel('N', fontdict=font)
    ax.legend(prop={'family': 'serif', 'size': 10})
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    return ax

def generate_figure():
    # Create figure and GridSpec layout
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, figure=fig)

    # Create subplots
    axes = [
        fig.add_subplot(gs[0, 0]),  # Top left
        fig.add_subplot(gs[0, 1]),  # Top right
        fig.add_subplot(gs[1, 0]),  # Middle left
        fig.add_subplot(gs[1, 1]),  # Middle right
        fig.add_subplot(gs[2, :]),  # Bottom spanning both columns
    ]

    plot_combined_R(axes[0])
    plot_combined_wavelengths(axes[1])
    plot_Flamda_for_disc4(axes[2])
    plot_disc_2_white(axes[3])
    plot_normalized(axes[4])

    # Annotate each subplot with a capital letter
    for i, ax in enumerate(axes):
        if i < 4:
            ax.annotate(
                f"{chr(65 + i)}",  # A, B, C, etc.
                xy=(-0.18, 0.95),  # Position near the top-left corner
                xycoords='axes fraction',
                fontsize=14,
                fontweight='bold',
                ha='left',
                va='top',)
        else:
            ax.annotate(
                f"{chr(65 + i)}",  # A, B, C, etc.
                xy=(-0.1, 0.95),  # Position near the top-left corner
                xycoords='axes fraction',
                fontsize=14,
                fontweight='bold',
                ha='left',
                va='top',
            )

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    #plt.savefig('graphs/isochromas_vs_force_big_subplot.png', dpi=300)

generate_figure()






