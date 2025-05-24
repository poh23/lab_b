import pandas as pd
import numpy as np
import PyMieScatt as PMS
import scipy.stats as stats

# === 2. Constants ===
n_sphere = 1.59
n_medium = 1.33
m_rel = n_sphere / n_medium
L = 0.8  # cm path length

def compute_Q_theory_pdf(diameter,diameter_error_um2, wl, n_bins=50):
    """
    Computes Q_sca averaged over a normal PDF of diameters.
    - mean diameter = row['diameter'] (μm)
    - std dev       = row['diameter_error_um2'] (μm)
    Returns the PDF-weighted average Q_sca.
    """
    # build an array of diameters in nm
    d_mean = diameter * 1e3             # μm → nm
    d_sigma = diameter_error_um2 * 2 * 1e3  # μm → nm
    # if sigma is zero, fall back to single value
    if d_sigma == 0:
        return PMS.MieQ(m=m_rel, wavelength=wl, diameter=d_mean)[1]
    # sample diameters ±3σ
    diameters = np.linspace(d_mean - 3*d_sigma,
                            d_mean + 3*d_sigma,
                            n_bins)

    pdf = stats.norm.pdf(diameters, loc=d_mean, scale=d_sigma)
    pdf /= np.trapz(pdf, diameters)  # normalize area = 1
    # compute Qsca for each diameter
    Qsca_vals = np.array([
        PMS.MieQ(m=m_rel,
                 wavelength=wl,
                 diameter=d)[1]
        for d in diameters
    ])
    # return weighted average
    return np.trapz(Qsca_vals * pdf, diameters)

def compute_cross_section(
    ref_file, meas_file,
    diameter_um, wavelength_nm,
    medium_index=1.33, particle_index=1.59,
    path_length_cm=0.8,
    m_susp=0.026, delta_m_susp=0.001,
    V_sol=3.5, delta_V_sol=0.1,
    suspension_concentration=0.025,  # 2.5% w/v
    rho_ps=1.05,  # g/cm³
    wavelength_error_nm=50,
    radius_error_um=0.25
):

    # Read data
    df_ref = pd.read_csv(ref_file)
    df_meas = pd.read_csv(meas_file)

    radius_um = diameter_um / 2

    # Get statistics
    I0 = df_ref['Ch0[V]'].mean()
    I0_std = df_ref['Ch0[V]'].std()
    I = df_meas['Ch0[V]'].mean()
    I_std = df_meas['Ch0[V]'].std()

    # Error in D from I and I0
    rel_error_I = I_std / I
    rel_error_I0 = I0_std / I0
    rel_error_D = np.sqrt(rel_error_I**2 + rel_error_I0**2)

    # Compute number density and its uncertainty
    r_cm = (diameter_um / 2) * 1e-4  # um to cm
    V_bead = (4/3) * np.pi * r_cm**3
    m_beads = m_susp * suspension_concentration
    V_beads = m_beads / rho_ps
    N_beads = V_beads / V_bead
    number_density = N_beads / V_sol

    # Compute uncertainty in number density using partial derivatives
    rel_error_rho = np.sqrt(
        (delta_m_susp / m_susp) ** 2 +
        (delta_V_sol / V_sol) ** 2 +
        (3 * (radius_error_um / (diameter_um / 2))) ** 2  # 3 * (Δr / r)
    )
    rho_err = number_density * rel_error_rho

    # Experimental cross section
    D = -np.log(I / I0)
    sigma_exp_cm2 = D / (number_density * path_length_cm)
    sigma_exp_um2 = sigma_exp_cm2 * 1e8
    geom_cross_section_um2 = np.pi * (radius_um ** 2)
    rel_geom_cross_section_um2_err = 2 * radius_error_um / radius_um
    Q_exp = sigma_exp_um2 / geom_cross_section_um2

    # Combine relative errors
    rel_error_Q = np.sqrt(rel_error_D**2 + rel_error_rho**2 + rel_geom_cross_section_um2_err**2)
    Q_exp_err = Q_exp * rel_error_Q

    # Theoretical Q using PyMieScatt
    m = particle_index / medium_index
    radius_um = diameter_um / 2
    Qsca = compute_Q_theory_pdf(diameter_um, radius_error_um*2, wavelength_nm)

    # Estimate theoretical error using finite differences

    # Error due to wavelength
    Q_up = compute_Q_theory_pdf(diameter_um, radius_error_um*2, wavelength_nm + wavelength_error_nm)
    Q_down = compute_Q_theory_pdf(diameter_um, radius_error_um*2, wavelength_nm - wavelength_error_nm)
    dQ_dlambda = (Q_up - Q_down) / (2 * wavelength_error_nm)

    Q_err = np.abs(dQ_dlambda * wavelength_error_nm)

    # Print results
    print(f"I0 = {I0:.4f} ± {I0_std:.4f}")
    print(f"I  = {I:.4f} ± {I_std:.4f}")
    print(f"D  = {D:.4f} ± {D * rel_error_D:.4f}")
    print(f"ρ  = {number_density:.2e} ± {rho_err:.2e} particles/cm³")
    print(f"\nExperimental Q = {Q_exp:.4e} ± {Q_exp_err:.2e}")
    print(f"Theoretical  Q = {Qsca:.4e} ± {Q_err:.2e}")
    print(f"Relative Q_exp / Q_theory = {Q_exp / Qsca:.4f}")
    print(f"T-test: {np.abs(Q_exp - Qsca) / np.sqrt(Q_exp_err**2 + Q_err**2):.4f}")

# Example usage:
compute_cross_section(
    ref_file="../data/Mie/ref_red_precise.csv",
    meas_file="../data/Mie/0.5r_red_precise.csv",
    diameter_um=0.5,
    wavelength_nm=640,
    wavelength_error_nm=20,
    radius_error_um=0.025,
)
