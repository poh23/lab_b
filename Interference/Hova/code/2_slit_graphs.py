import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

matplotlib.use('TkAgg')
# ---- Settings ----
font = {'family': 'serif', 'size': 12}
plt.rc('font', **font)

# ------------------------- Load and Prepare Data -------------------------
# c=0.09 for narrow slit
def load_data(file_path, m=9.62, c=0.05):
    df = pd.read_csv(file_path)
    df['Intensity'] = -df['Ch0[V]']
    df['Angle'] = m * df['Ch1[V]'] + c
    return df[['Angle', 'Intensity']]
# ------------------------- Sinc Squared Model -------------------------
def double_sinc_squared(angles, A, a_eff, d_eff):
    theta_rad = np.radians(angles)
    sinc_term = np.sinc(a_eff * np.sin(theta_rad))
    interference = np.cos(np.pi * d_eff * np.sin(theta_rad))**2
    return A * sinc_term**2 * interference + 0.05

# ------------------------- Double-Slit Convolved Model -------------------------
def double_slit_convolved(angle_data, A, a_eff, d_eff, dtheta=0.5):
    angle_fine = np.linspace(angle_data.min() - 1, angle_data.max() + 1, 5000)
    theta_rad = np.radians(angle_fine)
    sinc_term = np.sinc(a_eff * np.sin(theta_rad))
    interference = np.cos(np.pi * d_eff * np.sin(theta_rad))**2
    intensity_fine = A * sinc_term**2 * interference + 0.05

    delta = angle_fine[1] - angle_fine[0]
    window_size = max(1, int(dtheta / delta))
    intensity_smooth = uniform_filter1d(intensity_fine, size=window_size)

    interp_func = interp1d(angle_fine, intensity_smooth, bounds_error=False, fill_value=0)
    return interp_func(angle_data)

# ------------------------- Fitting and Plotting -------------------------
def fit_double_slit_convolved(df, ax_main, ax_resid, slit_type,init_a=80,init_d=40, aperture_width_deg=0.5, wavelength=632.8e-9):
    angle_data = df['Angle'].values
    intensity_data = df['Intensity'].values
    sigma_intensity = 0.005

    def model(angle, A, a_eff, d_eff):
        return double_slit_convolved(angle, A, a_eff, d_eff, dtheta=aperture_width_deg)

    popt, pcov = curve_fit(
        model,
        angle_data,
        intensity_data,
        p0=[max(intensity_data), init_a, init_d],
        sigma=np.full_like(intensity_data, sigma_intensity),
        absolute_sigma=True
    )

    A_fit, a_eff_fit, d_eff_fit = popt
    perr = np.sqrt(np.diag(pcov))
    sigma_A, sigma_a_eff, sigma_d_eff = perr

    slit_width = a_eff_fit * wavelength
    slit_spacing = d_eff_fit * wavelength
    slit_width_err = sigma_a_eff * wavelength
    slit_spacing_err = sigma_d_eff * wavelength

    model_fit = model(angle_data, *popt)
    residuals = intensity_data - model_fit
    chi2 = np.sum((residuals / sigma_intensity)**2)
    dof = len(angle_data) - len(popt)
    chi2_red = chi2 / dof

    # Plot
    angle_fit = np.linspace(angle_data.min(), angle_data.max(), 1000)
    intensity_fit = model(angle_fit, *popt)

    ax_main.errorbar(angle_data, intensity_data,
                     xerr=0.05, yerr=sigma_intensity,
                     fmt='o', markersize=2, ecolor='gray', elinewidth=0.5, capsize=0,
                     label='Experimental Data', alpha=0.8)
    ax_main.plot(angle_fit, intensity_fit, color='orange', linewidth=2, label='Double-Slit Fit')
    #ax_main.plot(angle_data, double_sinc_squared(angle_data, 6, 126.4, 39.5), color='orange', linewidth=2, label='Double-Slit Func (No Convolution)')
    ax_main.set_ylabel('Intensity [a.u.]')
    ax_main.set_title(f'{slit_type} Slits')
    ax_main.legend(fontsize=10)
    ax_main.grid(True, linestyle='--', linewidth=0.5)

    ax_resid.axhline(0, color='red', linestyle='--', linewidth=1)
    ax_resid.scatter(angle_data, residuals, s=10, color='black')
    ax_resid.set_ylabel('Residuals')
    ax_resid.set_xlabel(r'Angle [$\degree$]')
    ax_resid.grid(True, linestyle='--', linewidth=0.5)

    print(f"\n=== Convolved Double-Slit Fit Results ===")
    print(f"A = {A_fit:.3f} ± {sigma_A:.3f}")
    print(f"a_eff = {a_eff_fit:.3f} ± {sigma_a_eff:.3f}")
    print(f"d_eff = {d_eff_fit:.3f} ± {sigma_d_eff:.3f}")
    print(f"Slit width = {slit_width * 1e6:.2f} ± {slit_width_err * 1e6:.2f} µm")
    print(f"Slit spacing = {slit_spacing * 1e6:.2f} ± {slit_spacing_err * 1e6:.2f} µm")
    print(f"Reduced chi-squared = {chi2_red:.2f}")

    return ax_main, ax_resid

# ------------------------- Example Usage -------------------------
df_wide = load_data('../data/two_wideslit.csv', c =0.05)
df_narrow = load_data('../data/two_narrowslit.csv', c =0.09)

# Create subplots in 4 rows: 2 data + 2 residuals
fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex='col',
                        gridspec_kw={'height_ratios': [3, 1]})

(ax_main_narrow, ax_main_wide), (ax_res_narrow, ax_res_wide) = axs

# Analyze and plot narrow and wide slit data
fit_double_slit_convolved(
    df_narrow, ax_main_narrow, ax_res_narrow, init_a=20, init_d=40,
    slit_type='Narrow', aperture_width_deg=0.001
)

fit_double_slit_convolved(
    df_wide, ax_main_wide, ax_res_wide,
    slit_type='Wide', aperture_width_deg=0.001
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
plt.savefig('../graphs/two_slits_figure.png')
plt.show()