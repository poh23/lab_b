import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PyMieScatt as PMS
import matplotlib
matplotlib.use('TkAgg')


# === 1. Load your data ===
df = pd.read_csv("../data/Mie/consolidated_data.csv")

# === 2. Constants ===
n_sphere = 1.59
n_medium = 1.33
m_rel = n_sphere / n_medium
L = 0.8  # cm path length

density_map = {
    0.50: 3.6e9,
    0.75: 1.07e9,
    1.00: 4.5e8
}

df["density"] = df["diameter"].map(density_map) # added factor of 6 to match the theoretical values

# === 3. Add calculated fields ===
df["radius_um"] = df["diameter"] / 2
df["mx"] = 2 * np.pi * df["radius_um"] * m_rel / df["wavelength"]
df["sigma_exp_cm2"] = (-np.log(df["normalized"])) / (df["density"] * L)
df["sigma_exp_um2"] = df["sigma_exp_cm2"] * 1e8
df["geom_cross_section_um2"] = np.pi * (df["radius_um"] ** 2)
df["Q_exp"] = df["sigma_exp_um2"] / df["geom_cross_section_um2"]
df["Q_exp_err"] = np.sqrt((df["std_dev_ref"] / df["reference"]) ** 2 + (df["std_dev"] / df["mean"]) ** 2)/(df["density"] * L * 1e-8 * df["geom_cross_section_um2"])
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
        fmt=marker_map[row["diameter"]],
        color=color_map[row["wavelength"]],
        capsize=3, markersize=5,
        label=f'D={row["diameter"]}μm, λ={row["wavelength"]}nm'
    )

# Plot theoretical Qscattering
df = df.sort_values(by=["Q_theory"])
plt.plot(df["mx"], df["Q_theory"], color='black', label='Theoretical Qscattering (Mie)', zorder=10)

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

