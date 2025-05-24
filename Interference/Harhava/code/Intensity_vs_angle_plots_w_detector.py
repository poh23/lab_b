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

err_dict = {
    0.5: 0.05,
    0.75: 0.075,
    1: 0.1
}


# Load the CSV data
def read_csv_data():
    return pd.read_csv('../data/Mie_angles/consolidated_angle_data.csv', encoding='utf-8')


def compute_SU_avg(m, wavelength, diameter, angle_list, width):
    diameter_nm = diameter * 1000  # μm -> nm

    # Ensure we get a high-resolution angle calculation that spans the range we need
    margin = max(10, width + 2)
    min_angle = max(0.1, min(angle_list) - margin)
    max_angle = min(180, max(angle_list) + margin)

    theta_hr, SL, SR, SU = pm.ScatteringFunction(
        m, wavelength, diameter_nm,
        minAngle=min_angle, maxAngle=max_angle,
        angularResolution=0.1,
        normalization=None
    )
    theta_hr = np.degrees(theta_hr)
    SU_avg = []
    for theta in angle_list:
        mask = (theta_hr >= theta - width / 2) & (theta_hr <= theta + width / 2)
        avg = np.mean(SU[mask]) if np.any(mask) else 0
        SU_avg.append(avg)
    return np.array(SU_avg)



# Normalize the experimental data for fair comparison
def normalize_data(data_group):
    # Normalize relative to the first point (smallest angle)
    first_angle_idx = data_group['angle'].idxmin()
    first_value = data_group.loc[first_angle_idx, 'mean']

    if first_value != 0:
        data_group['normalized_mean'] = data_group['mean'] / first_value
        data_group['normalized_std'] = data_group['std_dev'] / abs(first_value)
    else:
        # Fallback if first value is zero
        data_group['normalized_mean'] = data_group['mean'] / data_group['mean'].abs().max()
        data_group['normalized_std'] = data_group['std_dev'] / data_group['mean'].abs().max()

    return data_group


# Main visualization function
def visualize_scattering_data(detector_width=5.0, lambda0 = 633, wavelength_error=5, n_particle=1.59, n_medium=1):
    """
    Create visualization with specified detector width

    Args:
        detector_width: Width of the detector in degrees (default: 5.0)
    """
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
    ax4.set_title(f'Combined Data for All Diameters\nDetector Width: {detector_width}°', fontsize=20)
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

        # Calculate theoretical values using PyMieScatt with high resolution
        angles = diameter_data['angle'].values
        # Compute central SU
        diameter_error = err_dict[diameter]
        m = n_particle / n_medium
        SU0 = compute_SU_avg(m, lambda0, diameter, angles, detector_width)

        # Numerical derivatives
        dl = wavelength_error
        dr = diameter_error

        SU_plus_l = compute_SU_avg(m, lambda0 + dl, diameter, angles, detector_width)
        SU_minus_l = compute_SU_avg(m, lambda0 - dl, diameter, angles, detector_width)
        df_dlambda = (SU_plus_l - SU_minus_l) / (2 * dl)

        SU_plus_r = compute_SU_avg(m, lambda0, diameter + dr, angles, detector_width)
        SU_minus_r = compute_SU_avg(m, lambda0, diameter - dr, angles, detector_width)
        df_dradius = (SU_plus_r - SU_minus_r) / (2 * dr)

        # Total propagated uncertainty
        delta_SU = np.sqrt((df_dlambda * dl) ** 2 + (df_dradius * dr) ** 2)

        # Normalize to forward value
        SU0_norm = SU0 / SU0[0]
        delta_norm = delta_SU / SU0[0]  # Propagated uncertainty normalized

        # Plot individual diameter subplot
        if i < 3:  # We have 3 diameter-specific subplots
            ax = axes[i]
            ax.errorbar(angles, diameter_data['normalized_mean'],
                        fmt=markers[i], xerr=angle_error, yerr=diameter_data['normalized_std'],
                        label=f'Measured (d={diameter}μm)', color=colors[i], markersize=5, alpha=0.7)
            ax.plot(angles, SU0_norm, 'k-', label='Normalized $f(\\theta)$')
            ax.fill_between(angles, SU0_norm - delta_norm, SU0_norm + delta_norm,
                             color='gray', alpha=0.4, label='Propagated uncertainty')

            ax.set_title(f'Diameter = {diameter}μm', fontsize=20)
            ax.set_xlabel(r'Angle [$\degree$]', fontsize=18)
            ax.set_ylabel('Normalized Intensity', fontsize=18)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=16)

        # Add to combined plot
        ax4.errorbar(angles, diameter_data['normalized_mean'],
                     fmt=markers[i], xerr=angle_error, yerr=diameter_data['normalized_std'],
                     label=f'Measured (d={diameter}μm)', color=colors[i], markersize=3, alpha=0.7)

    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=16)

    # Add subplot labels (A, B, C, D)
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

    # Modify the title to show the detector width
    fig.suptitle(f'Scattering Data Analysis (Detector Width: {detector_width}°)',
                 fontsize=22, y=0.99)
    plt.subplots_adjust(top=0.93)

    #plt.savefig(f'scattering_data_width_{detector_width}.pdf')
    plt.show()

    return fig



# Run the visualization with explicit detector width parameter
if __name__ == "__main__":
    # For a single visualization with default detector width of 5°
    visualize_scattering_data()
