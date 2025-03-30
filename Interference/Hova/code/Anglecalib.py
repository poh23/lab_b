import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'serif', 'size': 12}
plt.rc('font', **font)

# ---------------- Load the CSV ----------------
file_path = r'..\data\Angle voltCalibration_1.csv'
df_voltage = pd.read_csv(file_path)
df_voltage.columns = ["Angle", "Voltage"]

# ---------------- Define measurement errors ----------------
sigma_angle = 0.05    # degrees
sigma_voltage = 0.005 # volts

# ---------------- Linear Fit with polyfit + covariance ----------------
coeffs, cov = np.polyfit(df_voltage["Voltage"], df_voltage["Angle"], 1, cov=True)
m_fit, b_fit = coeffs
perr = np.sqrt(np.diag(cov))

# ---------------- Smooth Fit Line ----------------
voltage_range = np.linspace(df_voltage["Voltage"].min(), df_voltage["Voltage"].max(), 100)
angle_fit = np.polyval(coeffs, voltage_range)

# ---------------- Plot ----------------
plt.figure(figsize=(8, 5))

# Plot data with error bars
plt.errorbar(
    df_voltage["Voltage"],
    df_voltage["Angle"],
    xerr=sigma_voltage,
    yerr=sigma_angle,
    fmt='o',
    ecolor='gray',
    capsize=3,
    label='Measured Data'
)

# Plot the fit line
plt.plot(voltage_range, angle_fit, '-', label=f'Fit: Angle = ({m_fit:.2f} ± {perr[0]:.2f})·V + ({b_fit:.2f} ± {perr[1]:.2f})')

plt.xlabel("Voltage [V]")
plt.ylabel(r'Angle [$\degree$]')
plt.title("Angle Calibration with Measurement Errors")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(r'..\graphs\angle_calibration.png')

# ---------------- Report ----------------
print("=== Fit Results ===")
print(f"Slope (m) = {m_fit:.3f} ± {perr[0]:.3f} degrees/Volt")
print(f"Intercept (b) = {b_fit:.3f} ± {perr[1]:.3f} degrees")
