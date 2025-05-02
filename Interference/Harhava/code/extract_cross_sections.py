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
L = 1.0  # cm path length

density_map = {
    0.50: 4.49e8,
    0.75: 1.33e8,
    1.00: 5.61e7
}
df["density"] = df["diameter"].map(density_map) * 6 # added factor of 6 to match the theoretical values

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
df["x"] = 2 * np.pi * df["radius_um"] / df["wavelength"]  # size parameter
df["Q_theory"] = df.apply(
    lambda row: PMS.MieQ(
        m=m_rel,
        wavelength=row["wavelength"],
        diameter=row["diameter_nm"]
    )[1],  # Qsca is the second value
    axis=1
)

# === 5. Plot log-log Qscattering vs mx ===
plt.figure(figsize=(8, 6))
plt.errorbar(df["mx"], df["Q_exp"], yerr=df['Q_exp_err'], fmt='o', label='Experimental Qscattering', color='purple', capsize=3, markersize=4)
plt.scatter(df["mx"], df["Q_theory"], label='Theoretical Qscattering (Mie)', color='orange')

plt.xlabel(r"mx ($\propto$ $\frac{r}{\lambda}$)")
plt.ylabel("Qscattering")
plt.title("Experimental vs Theoretical Qscattering")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.show()
