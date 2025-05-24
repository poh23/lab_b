import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import PyMieScatt as pm
import matplotlib

matplotlib.use('TkAgg')

angle_error = 1  # Error in angle measurement

# ---- Settings ----
font = {'family': 'serif', 'size': 16}
plt.rc('font', **font)

# Load the CSV data
def read_csv_data():
    # For this example, we'll use the window.fs API to read the uploaded file
    return pd.read_csv('../data/Mie_angles/consolidated_angle_data.csv', encoding='utf-8')


# Calculate theoretical values using PieMyscatt
def calculate_theoretical(diameter, angles, wavelength=633, n_particle=1.59):
    """
    Calculate the theoretical scattering intensities for given parameters using piemyscatt

    Args:
        diameter: particle diameter in micrometers
        angles: list of angles in degrees
        wavelength: wavelength of light in meters
        n_medium: refractive index of medium
        n_particle: refractive index of particle

    Returns:
        normalized intensities for the given angles
    """
    # Calculate Mie scattering
    diameter_nm = diameter * 1e3  # Convert diameter to nm
    # PieMyScat uses θ as the polar angle measured from the forward direction
    # so we need to use 180-angle_deg for backscattering setup
    scattered_intensity = pm.ScatteringFunction(n_particle, wavelength, diameter_nm, minAngle=min(angles), maxAngle=max(angles), angularResolution=10, normalization='max')
    return scattered_intensity


# Normalize the experimental data for fair comparison
def normalize_data(data_group):
    # Min-max normalization of the mean values
    min_val = data_group['mean'].min()
    max_val = data_group['mean'].max()
    if max_val != min_val:
        data_group['normalized_mean'] = data_group['mean']/data_group['mean'].values[0]
        data_group['normalized_std'] = data_group['std_dev']/np.abs(data_group['mean'].values[0])
    else:
        data_group['normalized_mean'] = 1

    return data_group


# Main visualization function
def visualize_scattering_data():
    # Read the data
    df = read_csv_data()

    # Get unique diameters
    diameters = df['diameter'].unique()

    # Set up the figure with GridSpec for more control
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    axes = [ax1, ax2, ax3]
    colors = ['b', 'g', 'r']
    markers = ['o', 's', '^']

    # For plotting all data in the fourth subplot
    ax4.set_title(f'Combined Data for All Diameters', fontsize=20)
    ax4.set_xlabel(r'Angle [$\degree$]', fontsize=18)
    ax4.set_ylabel('Normalized Intensity', fontsize=18)

    # Process each diameter
    for i, diameter in enumerate(diameters):
        # Filter data for this diameter
        diameter_data = df[df['diameter'] == diameter].copy()

        # Sort by angle for smooth line plots
        diameter_data = diameter_data.sort_values('angle')

        # Normalize the experimental data
        diameter_data = normalize_data(diameter_data)

        # Calculate theoretical values
        angles = diameter_data['angle'].values
        angles2, sl, sr, theoretical_values = calculate_theoretical(diameter, angles)

        # Plot individual diameter subplot
        if i < 3:  # We have 3 diameter-specific subplots
            ax = axes[i]
            ax.errorbar(angles-20, diameter_data['normalized_mean'] - 0.1, fmt=markers[i], xerr=angle_error, yerr=diameter_data['normalized_std'],
                       label=f'Measured (d={diameter}μm)', color=colors[i], markersize=5, alpha=0.7)
            ax.plot(angles, theoretical_values, 'k-', linewidth=2, label='Theoretical')

            ax.set_title(f'Diameter = {diameter}μm', fontsize=20)
            ax.set_xlabel(r'Angle [$\degree$]', fontsize=18)
            ax.set_ylabel('Normalized Intensity', fontsize=18)
            ax.set_xlim(0, 130)
            ax.set_ylim(0, 1.5)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=16)

        # Add to combined plot
        ax4.errorbar(angles, diameter_data['normalized_mean'], fmt=markers[i], xerr=angle_error, yerr=diameter_data['normalized_std'],
                    label=f'Measured (d={diameter}μm)', color=colors[i], markersize=3, alpha=0.7)

    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=16)

    axs = [ax1, ax2, ax3, ax4]
    for i, ax in enumerate(axs):
        ax.annotate(
            f"{chr(65 + i)}",  # A, B, C, etc.
            xy=(-0.15, 1),  # Position near the top-left corner
            xycoords='axes fraction',
            fontsize=20,
            fontweight='bold',
            ha='left',
            va='top', )

    plt.tight_layout(w_pad=2, h_pad=4)
    plt.savefig('../graphs/scattering_data_combined.pdf')
    plt.show()

    return fig


# Run the visualization
visualize_scattering_data()