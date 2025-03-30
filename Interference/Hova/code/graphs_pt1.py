import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use('TkAgg')

# ------------------ Settings ------------------
font = {'family': 'serif', 'size': 12}
plt.rc('font', **font)


# ------------------ Model ------------------
def sinc_squared(angle, A, a_eff):
    angle_rad = np.radians(angle)
    x = np.pi * a_eff * np.sin(angle_rad)
    return A * (np.sinc(x / np.pi)) ** 2


def v_to_angle(voltage):
    # Calibration parameters (from previous polyfit)
    m = 11.86  # slope [deg/V]
    c = -0.7  # intercept [deg]
    return m * voltage + c


# ------------------ Main function ------------------
def create_slit_graph(slit_type='narrow'):
    # ----------- Load data -----------
    ns_data = pd.read_csv(f'../data/{slit_type}slit.csv')
    ns_data['Intensity'] = ns_data['Ch0[V]'] * -1 # Invert the intensity
    ns_data['Angle'] = v_to_angle(ns_data['Ch1[V]'])
    ns_data = ns_data.drop(columns=['Ch0[V]', 'Ch1[V]'])

    angle_data = ns_data['Angle'].values
    intensity_data = ns_data['Intensity'].values
    voltage_data = ns_data['Angle'].values / 11.86  # reverse-calculated for uncertainty propagation

    # ----------- Uncertainties -----------
    sigma_voltage = 0.005  # [V]
    sigma_intensity = 0.005  # [a.u.]
    sigma_m = 0.08  # Uncertainty in m (from your calibration fit)

    sigma_angle = np.sqrt(
        (voltage_data * sigma_m) ** 2 + (11.86 * sigma_voltage) ** 2)  # Propagated uncertainty in angle

    # ----------- Fit the model -----------
    popt, pcov = curve_fit(
        sinc_squared, angle_data, intensity_data,
        sigma=np.full_like(intensity_data, sigma_intensity),
        absolute_sigma=True,
        p0=[intensity_data.max(), 1]
    )
    A_fit, a_eff_fit = popt
    perr = np.sqrt(np.diag(pcov))
    sigma_A, sigma_a_eff = perr

    # ----------- Reduced chi-squared -----------
    residuals = intensity_data - sinc_squared(angle_data, *popt)
    chi2 = np.sum((residuals / sigma_intensity) ** 2)
    dof = len(angle_data) - len(popt)
    chi2_red = chi2 / dof

    # ----------- Slit width -----------
    wavelength = 632.8e-9  # [m]
    slit_width = a_eff_fit * wavelength
    slit_width_err = sigma_a_eff * wavelength

    # ----------- Print results -----------
    print(f"\n=== Fitted Parameters for {slit_type} slit ===")
    print(f"A = {A_fit:.3f} ± {sigma_A:.3f}")
    print(f"a_eff = {a_eff_fit:.3f} ± {sigma_a_eff:.3f}")
    print(f"Estimated Slit Width = {slit_width * 1e6:.2f} ± {slit_width_err * 1e6:.2f} µm")
    print(f"Chi-Squared = {chi2:.2f}")
    print(f"Reduced Chi-Squared = {chi2_red:.2f}")

    # ----------- Prepare smooth curve for plot -----------
    angle_fit = np.linspace(angle_data.min(), angle_data.max(), 1000)
    intensity_nominal = sinc_squared(angle_fit, A_fit, a_eff_fit)
    intensity_upper = sinc_squared(angle_fit, A_fit + sigma_A, a_eff_fit + sigma_a_eff)
    intensity_lower = sinc_squared(angle_fit, A_fit - sigma_A, a_eff_fit - sigma_a_eff)

    # ----------- Plot -----------
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))

    # Data points with smaller markers and thin error bars (no caps)
    axs.errorbar(
        angle_data,
        intensity_data,
        xerr=sigma_angle,
        yerr=sigma_intensity,
        fmt='o',
        markersize=2,  # smaller points
        ecolor='gray',
        elinewidth=0.5,  # thinner error bars
        capsize=0,  # remove caps
        alpha=0.8,  # slight transparency
        label='Experimental Data'
    )

    # Best-fit curve
    axs.plot(angle_fit, intensity_nominal, color='orange', linewidth=2, label='sinc² Fit')

    # Error band
    axs.fill_between(angle_fit, intensity_lower, intensity_upper, color='orange', alpha=0.3, label='Fit Uncertainty')

    # Labels and title
    axs.set_ylabel(r'Intensity [a.u.]', fontsize=14)
    axs.set_xlabel(r'Angle [$\degree$]', fontsize=14)
    axs.set_title(f'Intensity vs Angle for {slit_type} Slit', fontsize=16)
    axs.legend(fontsize=12)
    axs.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f'../graphs/{slit_type}_slit.png')
    plt.show()


# ------------------ Run ------------------
create_slit_graph('narrow')
