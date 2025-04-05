import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

matplotlib.use('TkAgg')

# ---- Settings ----
font = {'family': 'serif', 'size': 12}
plt.rc('font', **font)

# ---- Sinc model with shift ----
def sinc_squared(angle, A, a_eff):
    angle_rad = np.radians(angle)
    x = np.pi * a_eff * np.sin(angle_rad)
    return A * (np.sinc(x / np.pi))**2

# ---- Convolved sinc model ----
def sinc_convolved(angle_data, A, a_eff, dtheta=0.5):
    angle_fine = np.linspace(angle_data.min() - 1, angle_data.max() + 1, 5000)
    intensity_fine = sinc_squared(angle_fine, A, a_eff)

    delta = angle_fine[1] - angle_fine[0]
    window_size = max(1, int(dtheta / delta))
    intensity_smooth = uniform_filter1d(intensity_fine, size=window_size)

    interp_func = interp1d(angle_fine, intensity_smooth, bounds_error=False, fill_value=0)
    return interp_func(angle_data)

# ---- Main plotting and fitting function ----
def analyze_slit(filename, ax_main, ax_resid, slit_type='narrow', aperture_width_deg=0.5, m=11.86, c=-0.7):
    df = pd.read_csv(filename)
    df['Intensity'] = np.abs(df['Ch0[V]'])

    # Calibration: angle = m * V + c
    sigma_m = 0.10
    sigma_voltage = 0.005
    sigma_intensity = 0.005

    df['Angle'] = m * df['Ch1[V]'] + c
    V_measured = (df['Angle'] - c) / m
    sigma_angle = np.sqrt((V_measured * sigma_m)**2 + (m * sigma_voltage)**2)

    angle_data = df['Angle'].values
    intensity_data = df['Intensity'].values

    def model(angle, A, a_eff):
        return sinc_convolved(angle, A, a_eff, dtheta=aperture_width_deg)

    popt, pcov = curve_fit(
        model,
        angle_data,
        intensity_data,
        p0=[intensity_data.max(), 120],
        sigma=np.full_like(intensity_data, sigma_intensity),
        absolute_sigma=True
    )

    A_fit, a_eff_fit = popt
    perr = np.sqrt(np.diag(pcov))
    sigma_A, sigma_a_eff = perr

    wavelength = 632.8e-9
    slit_width = a_eff_fit * wavelength
    slit_width_err = sigma_a_eff * wavelength

    model_fit = model(angle_data, *popt)
    residuals = intensity_data - model_fit
    chi2 = np.sum((residuals / sigma_intensity)**2)
    dof = len(angle_data) - len(popt)
    chi2_red = chi2 / dof

    print(f"\n=== Fitted Parameters for {slit_type} slit ===")
    print(f"A = {A_fit:.3f} ± {sigma_A:.3f}")
    print(f"a_eff = {a_eff_fit:.3f} ± {sigma_a_eff:.3f}")
    print(f"Slit width = {slit_width*1e6:.2f} ± {slit_width_err*1e6:.2f} µm")
    print(f"Reduced chi-squared = {chi2_red:.2f}")

    # Fine curve for fit and band
    angle_fit = np.linspace(angle_data.min(), angle_data.max(), 1000)
    intensity_nominal = model(angle_fit, *popt)
    intensity_upper = model(angle_fit, A_fit + sigma_A, a_eff_fit + sigma_a_eff)
    intensity_lower = model(angle_fit, A_fit - sigma_A, a_eff_fit - sigma_a_eff)

    # Main plot
    ax_main.errorbar(angle_data, intensity_data,
                     xerr=sigma_angle, yerr=sigma_intensity,
                     fmt='o', markersize=2, ecolor='gray', elinewidth=0.5, capsize=0,
                     label='Experimental Data', alpha=0.8)
    ax_main.plot(angle_fit, intensity_nominal, color='orange', linewidth=2, label='sinc² Fit')
    ax_main.fill_between(angle_fit, intensity_lower, intensity_upper,
                         color='orange', alpha=0.3, label='Fit Uncertainty')
    ax_main.set_ylabel('Intensity [a.u.]')
    ax_main.set_title(f'{slit_type} Slit')
    ax_main.legend(fontsize=10)
    ax_main.grid(True, linestyle='--', linewidth=0.5)

    # Residual plot
    ax_resid.axhline(0, color='red', linestyle='--', linewidth=1)
    ax_resid.scatter(angle_data, residuals, s=10, color='black')
    ax_resid.set_ylabel('Residuals')
    ax_resid.set_xlabel(r'Angle [$\degree$]')
    ax_resid.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return ax_main, ax_resid

# Create subplots in 4 rows: 2 data + 2 residuals
fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex='col',
                        gridspec_kw={'height_ratios': [3, 1]})

(ax_main_narrow, ax_main_wide), (ax_res_narrow, ax_res_wide) = axs

# Analyze and plot narrow and wide slit data
analyze_slit(
    '../data/narrowslit.csv', ax_main_narrow, ax_res_narrow,
    slit_type='Narrow', aperture_width_deg=0.0001
)

analyze_slit(
    '../data/wideslit_new.csv', ax_main_wide, ax_res_wide,
    slit_type='Wide', aperture_width_deg=0.001,
    m=9.62, c=0.11
)

axs = axs.ravel()
for i, ax in enumerate(axs):
    ax.annotate(
        f"{chr(65 + i)}",  # A, B, C, etc.
        xy=(-0.15, 1),  # Position near the top-left corner
        xycoords='axes fraction',
        fontsize=14,
        fontweight='bold',
        ha='left',
        va='top', )

plt.tight_layout()
plt.savefig('../graphs/one_slit_figure.png')
plt.show()



