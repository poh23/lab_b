import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PyMieScatt as PMS
import matplotlib
matplotlib.use('TkAgg')


# ---- Settings ----
font = {'family': 'serif', 'size': 14}
plt.rc('font', **font)

# Estimate Q_theory error from radius and wavelength uncertainties
def compute_Q_theory_error(row):
    r_nom = row["radius_um"]
    wl_nom = row["wavelength"]
    d_nom = row["diameter_nm"]
    radius_error_um = row["radius_error_um"]
    wavelength_error_nm = row["wavelength_error_nm"]

    # Radius up/down
    r_up = r_nom + radius_error_um
    r_down = r_nom - radius_error_um
    d_up = r_up * 2 * 1e3
    d_down = r_down * 2 * 1e3

    Q_r_up = PMS.MieQ(m=m_rel, wavelength=wl_nom, diameter=d_up)[1]
    Q_r_down = PMS.MieQ(m=m_rel, wavelength=wl_nom, diameter=d_down)[1]
    dQ_dr = (Q_r_up - Q_r_down) / (2 * radius_error_um)

    # Wavelength up/down
    wl_up = wl_nom + wavelength_error_nm
    wl_down = wl_nom - wavelength_error_nm

    Q_wl_up = PMS.MieQ(m=m_rel, wavelength=wl_up, diameter=d_nom)[1]
    Q_wl_down = PMS.MieQ(m=m_rel, wavelength=wl_down, diameter=d_nom)[1]
    dQ_dwl = (Q_wl_up - Q_wl_down) / (2 * wavelength_error_nm)

    # Combine uncertainties
    Q_err = np.sqrt((dQ_dr * radius_error_um) ** 2 + (dQ_dwl * wavelength_error_nm) ** 2)
    return Q_err


def compute_density_and_error(
    diameter_um, diameter_err_um,
    V_susp_ml=0.035, V_susp_err_ml=0.015,
    V_water_ml=3.5, V_water_err_ml=0.5,
    wv_concentration=0.025, # g/ml
    rho_ps=1.05 # g/cm³
):
    m_beads_g = V_susp_ml * wv_concentration
    m_beads_err_g = V_susp_err_ml * wv_concentration

    V_beads_cm3 = m_beads_g / rho_ps
    V_beads_err_cm3 = m_beads_err_g / rho_ps

    r_cm = (diameter_um / 2) * 1e-4
    r_err_cm = (diameter_err_um / 2) * 1e-4

    V_bead_cm3 = (4/3) * np.pi * r_cm**3
    V_bead_err_cm3 = 4 * np.pi * r_cm**2 * r_err_cm

    N_beads = V_beads_cm3 / V_bead_cm3
    N_beads_err = N_beads * np.sqrt(
        (V_beads_err_cm3 / V_beads_cm3)**2 +
        (V_bead_err_cm3 / V_bead_cm3)**2
    )

    V_total_cm3 = V_susp_ml + V_water_ml
    V_total_err_cm3 = np.sqrt(V_susp_err_ml**2 + V_water_err_ml**2)

    rho = N_beads / V_total_cm3
    rho_err = rho * np.sqrt(
        (N_beads_err / N_beads)**2 +
        (V_total_err_cm3 / V_total_cm3)**2
    )
    print(f"for diameter {diameter_um} - rho: {rho:.2e} ± {rho_err:.2e} g/cm^3")

    return rho, rho_err

# === 1. Load your data ===
df = pd.read_csv("../data/Mie/consolidated_data.csv")

# === 2. Constants ===
n_sphere = 1.59
n_medium = 1.33
m_rel = n_sphere / n_medium
L = 0.8  # cm path length

# density per cm^2 for different diameters of polystyrene spheres
# Calculate density and error for each row
densities = []
density_errors = []

for _, row in df.iterrows():
    rho, rho_err = compute_density_and_error(
        diameter_um=row["diameter"],
        diameter_err_um=row["diameter_error_um2"]
    )
    densities.append(rho)
    density_errors.append(rho_err)

df["density"] = densities
df["density_err"] = density_errors
df["rel_density_err"] = df["density_err"] / df["density"]

# === 3. Add calculated fields ===
df["radius_um"] = df["diameter"] / 2
df["radius_error_um"] = df["diameter_error_um2"] / 2
df["mx"] = 2 * np.pi * df["radius_um"] * n_sphere / df["wavelength"]
df["mx_error"] = 2 * np.pi * df["radius_error_um"] * n_sphere / df["wavelength"]
df["sigma_exp_cm2"] = (-np.log(df["normalized"])) / (df["density"] * L)
df["sigma_exp_um2"] = df["sigma_exp_cm2"] * 1e8
df["geom_cross_section_um2"] = np.pi * (df["radius_um"] ** 2)
df["rel_geom_cross_section_um2_err"] = 2 * df["radius_error_um"] / df["radius_um"]
df["Q_exp"] = df["sigma_exp_um2"] / df["geom_cross_section_um2"]
df["rel_Q_exp_err"] = np.sqrt((df["std_dev_ref"] / df["reference"]) ** 2 + (df["std_dev"] / df["mean"]) ** 2 + (df["rel_density_err"]) ** 2 + (df["rel_geom_cross_section_um2_err"]) ** 2)
df["Q_exp_err"] = df["Q_exp"] * df["rel_Q_exp_err"]
df["diameter_nm"] = df["diameter"] * 1e3  # Convert diameter to nm

# === 4. Calculate theoretical Qsca using miepython ===
df["Q_theory"] = df.apply(
    lambda row: PMS.MieQ(
        m=m_rel,
        wavelength=row["wavelength"],
        diameter=row["diameter_nm"]
    )[1],  # Qsca is the second value
    axis=1
)

# Add theoretical Q error column
df["Q_theory_err"] = df.apply(compute_Q_theory_error, axis=1)

# === 5. Plot with different shapes for diameters and colors for wavelengths ===
plt.figure(figsize=(9, 6))

# Define unique markers and colors
markers = ['o', 's', 'v', '^', 'D', 'P', '*']
wavelengths = sorted(df["wavelength"].unique())
diameters = sorted(df["diameter"].unique())
colors = ['blue', 'green', 'red']
marker_map = dict(zip(diameters, markers))
color_map = dict(zip(wavelengths, colors))

# Plot each point with matching marker (by diameter) and color (by wavelength)
for _, row in df.iterrows():
    plt.errorbar(
        row["mx"], row["Q_exp"],
        yerr=row["Q_exp_err"],
        xerr=row["mx_error"],
        fmt=marker_map[row["diameter"]],
        color=color_map[row["wavelength"]],
        capsize=3, markersize=5,
        alpha=0.6,
        label=f'D={row["diameter"]}μm, λ={row["wavelength"]}nm'
    )

# Plot theoretical Qscattering
df = df.sort_values(by=["Q_theory"])
plt.plot(df["mx"], df["Q_theory"], color='black', label='Theoretical Qscattering (Mie)', zorder=10)
plt.fill_between(df["mx"], df["Q_theory"] - df["Q_theory_err"], df["Q_theory"] + df["Q_theory_err"],
                 color='gray', alpha=0.3, label='Theoretical ± error', zorder=5)

# Create custom legends
from matplotlib.lines import Line2D

# Marker legend for diameters
marker_legend = [Line2D([0], [0], marker=marker_map[d], color='k', linestyle='', label=f'{d} μm', markersize=6)
                 for d in diameters]

# Color legend for wavelengths
color_legend = [Line2D([0], [0], marker='o', color=color_map[w], linestyle='', label=f'{w} nm', markersize=6)
                for w in wavelengths]

plt.xlabel(r"mx ($\propto$ $\frac{r}{\lambda}$)")
plt.ylabel(r"$Q_{scattering}$")
#plt.title("Experimental vs Theoretical Qscattering")
plt.grid(True, which="both", ls="--")
plt.legend(handles=marker_legend + color_legend + [
    Line2D([0], [0], linestyle='-', color="black", label='Theoretical', markersize=6)
], loc='best')
plt.tight_layout()
plt.savefig("../graphs/experimental_vs_theoretical_Qscattering.pdf")
plt.show()

